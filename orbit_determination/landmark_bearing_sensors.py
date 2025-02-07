import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple

import brahe
import numpy as np
from brahe import Epoch, R_EARTH
from scipy.spatial.transform import Rotation

from image_simulation.earth_vis import EarthImageSimulator
from utils.earth_utils import lat_lon_to_ecef
from vision_inference.camera import Frame
from vision_inference.ml_pipeline import MLPipeline


class LandmarkBearingSensor(ABC):
    """
    Abstract class for a landmark bearing sensor, which inputs the satellite pose and outputs landmark bearing measurements.
    """

    @abstractmethod
    def take_measurement(
        self, epoch: Epoch, cubesat_position: np.ndarray, R_body_to_eci: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a landmark bearing measurement using the sensor.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param R_body_to_eci: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        pass


class RandomLandmarkBearingSensor(LandmarkBearingSensor):
    """
    A sensor that randomly generates landmark bearing measurements within a cone centered about the camera's boresight.
    """

    def __init__(self, config, max_measurements: int = 10, fov: float = np.deg2rad(120)):
        """
        :param config: The configuration dictionary.
        :param fov: The field of view of the camera in radians.
        :param max_measurements: The number of measurements to attempt to take at once. The actual number may be less.
        """
        camera_params = config["satellite"]["camera"]
        self.R_camera_to_body = Rotation.from_quat(
            np.asarray(camera_params["orientation_in_cubesat_frame"]), scalar_first=True
        ).as_matrix()
        self.t_body_to_camera = np.asarray(
            camera_params["position_in_cubesat_frame"]
        )  # in the body frame

        self.max_measurements = max_measurements
        self.fov = fov
        self.cos_fov = np.cos(fov)

    def sample_bearing_unit_vectors(self) -> np.ndarray:
        """
        Sample self.max_measurements random bearing unit vectors in the body frame that are within the camera's field
        of view, which is a cone centered about the camera's boresight.

        :return: A numpy array of shape (self.max_measurements, 3) containing the sampled bearing unit vectors in the body frame.
        """
        phi = 2 * np.pi * np.random.random(self.max_measurements)
        # uniformly sample cos(theta) instead of theta to get a uniform distribution on the unit sphere
        theta = np.arccos(np.random.uniform(self.cos_fov, 1, self.max_measurements))
        bearing_unit_vectors_cf = Rotation.from_euler("ZX", np.column_stack((phi, theta))).apply(
            np.array([0, 0, 1])
        )

        # sanity check
        assert np.all(bearing_unit_vectors_cf[:, 2] > self.cos_fov)

        bearing_unit_vectors_body = (self.R_camera_to_body @ bearing_unit_vectors_cf.T).T
        return bearing_unit_vectors_body

    @staticmethod
    def get_ray_and_earth_intersections(
        ray_dirs: np.ndarray, ray_start: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the intersection points of rays with the Earth.
        The input number of rays, N, the output number of intersection points, M,
        and the returned boolean array, valid_intersections, are related as follows:
        M == np.sum(valid_intersections) <= N.

        :param ray_dirs: A numpy array of shape (N, 3) containing the direction vectors of the rays in ECI coordinates.
                         Note that the direction vectors must all be normalized.
        :param ray_start: A numpy array of shape (3,) containing the starting point of the rays in ECI coordinates.
        :return: A tuple containing a boolean array  of shape (N,) indicating which rays intersected the Earth,
                 and a numpy array of shape (M, 3) containing the intersection points in ECI coordinates.
        """
        assert np.allclose(np.linalg.norm(ray_dirs, axis=1), 1), "ray_dirs must be normalized"

        # As = np.sum(ray_dirs ** 2, axis=1)  # this is always 1 since the rays are normalized
        Bs = 2 * ray_dirs @ ray_start
        C = np.sum(ray_start**2) - R_EARTH**2
        assert C > 0, "The ray start location is inside the Earth!"

        discriminants = Bs**2 - 4 * C

        """
        Since C > 0 and np.all(As > 0), if the roots are real they must have the same sign.
        Bs < 0 implies that the slope at x = 0 is negative, so the roots are positive.
        Intuitively, this check is equivalent to np.dot(ray_dir, ray_start) < 0 which checks if ray_dir is in
        the half-space that is pointing towards the Earth.
        """
        valid_intersections = (discriminants >= 0) & (Bs < 0)

        # pick the smaller of the two positive roots from the quadratic formula, since it is closer to the camera
        ts = (-Bs[valid_intersections] - np.sqrt(discriminants[valid_intersections])) / 2
        intersection_points = ray_start + ts[:, np.newaxis] * ray_dirs[valid_intersections, :]

        assert intersection_points.shape[0] == np.sum(valid_intersections)
        return valid_intersections, intersection_points

    def take_measurement(
        self, _: Epoch, cubesat_position_eci: np.ndarray, R_body_to_eci: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.
        The number of measurements, N, will be some number less than or equal to self.max_measurements.

        :param _: The epoch as an instance of brahe's Epoch class. Not used.
        :param cubesat_position_eci: The position of the satellite in ECI as a numpy array of shape (3,).
        :param R_body_to_eci: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        bearing_unit_vectors_body = self.sample_bearing_unit_vectors()
        bearing_unit_vectors_eci = (R_body_to_eci @ bearing_unit_vectors_body.T).T
        camera_position_eci = cubesat_position_eci + R_body_to_eci @ self.t_body_to_camera

        valid_intersections, landmark_positions_eci = self.get_ray_and_earth_intersections(
            bearing_unit_vectors_eci, camera_position_eci
        )
        bearing_unit_vectors_body = bearing_unit_vectors_body[valid_intersections, :]

        # sanity check
        for bearing_unit_vector_body, landmark_position_eci in zip(
            bearing_unit_vectors_body, landmark_positions_eci
        ):
            true_bearing_unit_vector_eci = landmark_position_eci - cubesat_position_eci
            true_bearing_unit_vector_eci /= np.linalg.norm(true_bearing_unit_vector_eci)

            assert np.allclose(
                true_bearing_unit_vector_eci, R_body_to_eci @ bearing_unit_vector_body
            )

        return bearing_unit_vectors_body, landmark_positions_eci


class SimulatedMLLandmarkBearingSensor(LandmarkBearingSensor):
    """
    A sensor that simulates an image of the Earth from the camera's pose and runs the ML pipeline to generate landmark bearing measurements.
    """

    def __init__(self, config):
        """
        :param config: The configuration dictionary.
        """
        camera_params = config["satellite"]["camera"]
        self.R_camera_to_body = Rotation.from_quat(
            np.asarray(camera_params["orientation_in_cubesat_frame"]), scalar_first=True
        ).as_matrix()
        self.t_body_to_camera = np.asarray(
            camera_params["position_in_cubesat_frame"]
        )  # in the body frame

        self.ml_pipeline = MLPipeline()
        self.earth_image_simulator = EarthImageSimulator()

    def take_measurement(
        self, epoch: Epoch, cubesat_position: np.ndarray, R_body_to_eci: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param R_body_to_eci: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        R_eci_to_ecef = brahe.frames.rECItoECEF(epoch)
        R_body_to_ecef = R_eci_to_ecef @ R_body_to_eci
        position_ecef = R_eci_to_ecef @ cubesat_position + R_body_to_ecef @ self.t_body_to_camera
        R_camera_to_ecef = R_body_to_ecef @ self.R_camera_to_body

        print(f"Taking measurement at {epoch=}, {cubesat_position=}, {R_body_to_eci=}")

        # simulate image
        image = self.earth_image_simulator.simulate_image(position_ecef, R_camera_to_ecef)

        if np.all(image == 0):
            print("No image detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        # run the ML pipeline on the image
        frame = Frame(image, 0, datetime.now())
        # TODO: queue requests to the model and send them in batches as the sim runs
        regions_and_landmarks = self.ml_pipeline.run_ml_pipeline_on_single(frame)
        if regions_and_landmarks is None:
            print("No salient regions detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        # save the image with the detected landmarks
        epoch_str = str(epoch).replace(":", "_").replace(" ", "_").replace(".", "_")
        output_dir = os.path.abspath(
            os.path.join(__file__, f"../log/simulated_images/seed_69420_epoch_{epoch_str}/")
        )
        os.makedirs(output_dir, exist_ok=True)
        self.ml_pipeline.visualize_landmarks(frame, regions_and_landmarks, output_dir)

        landmark_positions_ecef = np.zeros(shape=(0, 3))
        pixel_coordinates = np.zeros(shape=(0, 2))
        confidence_scores = np.zeros(shape=(0,))

        for region, landmarks in regions_and_landmarks:
            centroids_ecef = lat_lon_to_ecef(landmarks.centroid_latlons[np.newaxis, ...]).reshape(
                -1, 3
            )

            landmark_positions_ecef = np.concatenate(
                (landmark_positions_ecef, centroids_ecef), axis=0
            )
            pixel_coordinates = np.concatenate((pixel_coordinates, landmarks.centroid_xy), axis=0)
            confidence_scores = np.concatenate(
                (confidence_scores, landmarks.confidence_scores), axis=0
            )

        if len(confidence_scores) == 0:
            print("No landmarks detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        landmark_positions_eci = (R_eci_to_ecef.T @ landmark_positions_ecef.T).T
        bearing_unit_vectors_cf = self.earth_image_simulator.camera.pixel_to_bearing_unit_vector(
            pixel_coordinates
        )
        bearing_unit_vectors_body = (self.R_camera_to_body @ bearing_unit_vectors_cf.T).T

        print(f"Detected {len(landmark_positions_eci)} landmarks")

        # TODO: output confidence_scores too
        return bearing_unit_vectors_body, landmark_positions_eci

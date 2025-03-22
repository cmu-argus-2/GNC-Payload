"""
Module that manages the different landmark bearing sensors.
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple

import brahe
import numpy as np
from brahe import R_EARTH, Epoch
from scipy.spatial.transform import Rotation

# pylint: disable=import-error
from image_simulation.earth_vis import EarthImageSimulator
from sensors.camera_model import CameraModel
from utils.config_utils import load_config
from utils.earth_utils import lat_lon_to_ecef
from vision_inference.frame import Frame
from vision_inference.landmark_detector import LandmarkDetector
from vision_inference.ml_pipeline import MLPipeline


# pylint: disable=too-few-public-methods
class LandmarkBearingSensor(ABC):
    """
    Abstract class for a landmark bearing sensor, which inputs the satellite pose and outputs
    landmark bearing measurements.
    """

    @abstractmethod
    def take_measurement(
        self,
        epoch: Epoch,
        cubesat_position: np.ndarray,
        eci_R_body: np.ndarray,
        camera_model: CameraModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a landmark bearing measurement using the sensor.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param eci_R_body: The rotation matrix from the body frame to the ECI frame as a numpy
        array of shape (3, 3).
        :param camera_model: The camera model to use for the measurement.
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors
        in the body frame and a numpy array of shape (N, 3) containing the landmark positions in
        ECI coordinates.
        """
        pass


class RandomLandmarkBearingSensor(LandmarkBearingSensor):
    """
    A sensor that randomly generates landmark bearing measurements within a cone centered about the camera's boresight.
    """

    def __init__(self, max_measurements: int = 10, fov: float = np.deg2rad(120)) -> None:
        """
        :param fov: The field of view of the camera in radians.
        :param max_measurements: The number of measurements to attempt to take at once. The actual number may be less.
        """
        self.max_measurements = max_measurements
        self.fov = fov
        self.cos_fov = np.cos(fov)

    def sample_bearing_unit_vectors(self, camera_model: CameraModel) -> np.ndarray:
        """
        Sample self.max_measurements random bearing unit vectors in the body frame that are within the camera's field
        of view, which is a cone centered about the camera's boresight.

        :param camera_model: The camera model to use for the measurement.
        :return: A numpy array of shape (self.max_measurements, 3) containing the sampled bearing unit vectors in the
        body frame.
        """
        phi = 2 * np.pi * np.random.random(self.max_measurements)
        # uniformly sample cos(theta) instead of theta to get a uniform distribution on the unit sphere
        theta = np.arccos(np.random.uniform(self.cos_fov, 1, self.max_measurements))
        bearing_unit_vectors_cf = Rotation.from_euler("ZX", np.column_stack((phi, theta))).apply(
            np.array([0, 0, 1])
        )

        # sanity check
        assert np.all(bearing_unit_vectors_cf[:, 2] > self.cos_fov)

        bearing_unit_vectors_body = (camera_model.body_R_camera @ bearing_unit_vectors_cf.T).T
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
        bs = 2 * ray_dirs @ ray_start
        C = np.sum(ray_start**2) - R_EARTH**2
        assert C > 0, "The ray start location is inside the Earth!"

        discriminants = bs**2 - 4 * C
        # pylint: disable=pointless-string-statement
        """
        Since C > 0 and np.all(As > 0), if the roots are real they must have the same sign.
        bs < 0 implies that the slope at x = 0 is negative, so the roots are positive.
        Intuitively, this check is equivalent to np.dot(ray_dir, ray_start) < 0 which checks if ray_dir is in
        the half-space that is pointing towards the Earth.
        """
        valid_intersections = (discriminants >= 0) & (bs < 0)

        # pick the smaller of the two positive roots from the quadratic formula, since it is closer to the camera
        ts = (-bs[valid_intersections] - np.sqrt(discriminants[valid_intersections])) / 2
        intersection_points = ray_start + ts[:, np.newaxis] * ray_dirs[valid_intersections, :]

        assert intersection_points.shape[0] == np.sum(valid_intersections)
        return valid_intersections, intersection_points

    def take_measurement(
        self,
        _: Epoch,
        cubesat_position: np.ndarray,
        eci_R_body: np.ndarray,
        camera_model: CameraModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.
        The number of measurements, N, will be some number less than or equal to self.max_measurements.

        :param _: The epoch as an instance of brahe's Epoch class. Not used.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param eci_R_body: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :param camera_model: The camera model to use for the measurement.
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        bearing_unit_vectors_body = self.sample_bearing_unit_vectors(camera_model)
        bearing_unit_vectors_eci = (eci_R_body @ bearing_unit_vectors_body.T).T
        camera_position_eci = camera_model.get_camera_position(cubesat_position, eci_R_body)

        valid_intersections, landmark_positions_eci = self.get_ray_and_earth_intersections(
            bearing_unit_vectors_eci, camera_position_eci
        )
        bearing_unit_vectors_body = bearing_unit_vectors_body[valid_intersections, :]

        # sanity check
        for bearing_unit_vector_body, landmark_position_eci in zip(
            bearing_unit_vectors_body, landmark_positions_eci
        ):
            true_bearing_unit_vector_eci = landmark_position_eci - cubesat_position
            true_bearing_unit_vector_eci /= np.linalg.norm(true_bearing_unit_vector_eci)

            assert np.allclose(true_bearing_unit_vector_eci, eci_R_body @ bearing_unit_vector_body)

        return bearing_unit_vectors_body, landmark_positions_eci


class GroundTruthLandmarkBearingSensor(LandmarkBearingSensor):
    """
    A sensor that outputs the ground truth landmark bearing to all salient landmarks within a cone centered
    about the camera's boresight.
    Note that this DOES NOT (yet) accurately simulate the camera's field of view.
    """

    def __init__(self, fov: float = np.deg2rad(100)) -> None:
        self.fov = fov
        self.cos_fov_on_2 = np.cos(fov / 2)
        self.region_landmarks_ecef = GroundTruthLandmarkBearingSensor.load_region_landmark_ecef()

    @staticmethod
    def load_region_landmark_ecef() -> dict[str, np.ndarray]:
        """
        Load the ECEF coordinates of the landmarks from the CSV files for all salient regions.

        :return: A dictionary mapping region identifiers to numpy array of shape (N, 3) containing
                 the coordinates of the landmarks in ECEF.
        """
        salient_regions: List[str] = load_config()["vision"]["salient_mgrs_region_ids"]
        region_landmarks_ecef = {}
        for region_id in salient_regions:
            region_landmarks_csv = os.path.join(
                LandmarkDetector.MODEL_DIR,
                f"{region_id}/{region_id}_top_salient.csv",
            )
            region_landmarks = np.loadtxt(region_landmarks_csv, delimiter=",", skiprows=1)
            # TODO: change this to :2 once the lat and lon columns are reordered in the csvs
            region_landmarks_ecef[region_id] = lat_lon_to_ecef(region_landmarks[:, 1::-1])
        return region_landmarks_ecef

    # pylint: disable=too-many-locals
    def take_measurement(
        self,
        epoch: Epoch,
        cubesat_position: np.ndarray,
        eci_R_body: np.ndarray,
        camera_model: CameraModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param eci_R_body: The rotation matrix from the body frame to ECI as a numpy array of shape (3, 3).
        :param camera_model: The camera model to use for the measurement.
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        ecef_R_eci = brahe.frames.rECItoECEF(epoch)
        position_ecef = ecef_R_eci @ cubesat_position
        ecef_R_body = ecef_R_eci @ eci_R_body

        camera_axis_ecef = camera_model.get_camera_axis(ecef_R_body)
        camera_position_ecef = camera_model.get_camera_position(position_ecef, ecef_R_body)

        # TODO: optimize this by using the MGRS regions to filter out landmarks that are definitely not visible
        all_landmarks_ecef = np.concatenate(list(self.region_landmarks_ecef.values()), axis=0)

        is_same_hemisphere = all_landmarks_ecef @ camera_position_ecef > 0
        hemisphere_landmarks_ecef = all_landmarks_ecef[is_same_hemisphere, :]

        bearing_vectors_ecef = hemisphere_landmarks_ecef - camera_position_ecef
        bearing_unit_vectors_ecef = bearing_vectors_ecef / np.linalg.norm(
            bearing_vectors_ecef, axis=1, keepdims=True
        )

        is_visible = bearing_unit_vectors_ecef @ camera_axis_ecef > self.cos_fov_on_2
        visible_landmarks_ecef = hemisphere_landmarks_ecef[is_visible, :]
        visible_landmarks_eci = (ecef_R_eci.T @ visible_landmarks_ecef.T).T

        bearing_unit_vectors_body = (ecef_R_body.T @ bearing_unit_vectors_ecef[is_visible, :].T).T
        return bearing_unit_vectors_body, visible_landmarks_eci


class SimulatedMLLandmarkBearingSensor(LandmarkBearingSensor):
    """
    A sensor that simulates an image of the Earth from the camera's pose and runs the ML pipeline to generate landmark
    bearing measurements.
    """

    def __init__(self) -> None:
        """
        Initialize this SimulatedMLLandmarkBearingSensor.
        """
        self.ml_pipeline = MLPipeline()
        self.earth_image_simulator = EarthImageSimulator()

    # pylint: disable=too-many-locals
    def take_measurement(
        self,
        epoch: Epoch,
        cubesat_position: np.ndarray,
        eci_R_body: np.ndarray,
        camera_model: CameraModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param eci_R_body: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :param camera_model: The camera model to use for the measurement.
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        print(f"Taking measurement at {epoch=}, {cubesat_position=}, {eci_R_body=}")

        ecef_R_eci = brahe.frames.rECItoECEF(epoch)
        position_ecef = ecef_R_eci @ cubesat_position
        ecef_R_body = ecef_R_eci @ eci_R_body

        # simulate image
        frame = self.earth_image_simulator.simulate_image(position_ecef, ecef_R_body, camera_model)

        if np.all(frame.image == 0):
            print("No image detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        # run the ML pipeline on the image
        # TODO: queue requests to the model and send them in batches as the sim runs
        landmark_detections, region_slices = self.ml_pipeline.run_ml_pipeline_on_single(frame)

        # save the image with the detected landmarks
        epoch_str = str(epoch).replace(":", "_").replace(" ", "_").replace(".", "_")
        output_dir = os.path.abspath(
            os.path.join(__file__, f"../log/simulated_images/seed_69420_epoch_{epoch_str}/")
        )
        os.makedirs(output_dir, exist_ok=True)
        MLPipeline.visualize_landmarks(frame, landmark_detections, region_slices, output_dir)

        if len(region_slices) is None:
            print("No salient regions detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))
        if len(landmark_detections) == 0:
            print("No landmarks detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        landmark_positions_ecef = lat_lon_to_ecef(landmark_detections.latlons)
        landmark_positions_eci = (ecef_R_eci.T @ landmark_positions_ecef.T).T
        bearing_unit_vectors_cf = camera_model.pixel_to_bearing_unit_vector(
            landmark_detections.pixel_coordinates
        )
        bearing_unit_vectors_body = (camera_model.body_R_camera @ bearing_unit_vectors_cf.T).T

        print(f"Detected {len(landmark_positions_eci)} landmarks")

        # TODO: output confidences too
        return bearing_unit_vectors_body, landmark_positions_eci

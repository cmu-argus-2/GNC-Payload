from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from typing import Tuple
from time import perf_counter
from time import time
import yaml
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

import brahe
from brahe.constants import R_EARTH
from brahe.epoch import Epoch

from dynamics.orbital_dynamics import f
from image_simulation.earth_vis import EarthImageSimulator
from orbit_determination.nonlinear_least_squares_od import OrbitDetermination
from utils.earth_utils import lat_lon_to_ecef

from utils.orbit_utils import get_sso_orbit_state, is_over_daytime
from utils.brahe_utils import increment_epoch
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


class SimulatedMLLandmarkBearingSensor:
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


def load_config() -> dict[str, Any]:
    """
    Load the configuration file and modify it for the purposes of this test.

    :return: The modified configuration file as a dictionary.
    """
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # TODO: move this into the config file itself
    # decrease world update rate since we only care about position dynamics
    config["solver"]["world_update_rate"] = 1 / 60  # Hz
    config["mission"]["duration"] = 3 * 90 * 60  # s, roughly 1 orbit

    return config


# TODO: consolidate this with the function in utils/earth_utils.py
def get_nadir_rotation(cubesat_position: np.ndarray) -> np.ndarray:
    """
    Get the rotation matrix from the body frame to the ECI frame for a satellite with an orbital angular momentum in the -y direction.

    :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
    :return: A numpy array of shape (3, 3) containing the rotation matrix from the body frame to the ECI frame.
    """
    y_axis = [0, -1, 0]  # along orbital angular momentum
    z_axis = -cubesat_position / np.linalg.norm(cubesat_position)  # along radial vector
    x_axis = np.cross(y_axis, z_axis)
    R_body_to_eci = np.column_stack([x_axis, y_axis, z_axis])
    return R_body_to_eci


def get_SO3_noise_matrices(N: int, magnitude_std: float) -> np.ndarray:
    """
    Generate a set of matrices representing random rotations in SO(3) with a given standard deviation.

    :param N: The number of noise matrices to generate.
    :param magnitude_std: The standard deviation of the magnitudes of the rotations in radians.
    :return: A numpy array of shape (N, 3, 3) containing the noise rotations.
    """
    magnitudes = np.abs(np.random.normal(scale=magnitude_std, size=N))
    directions = np.random.normal(size=(N, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return Rotation.from_rotvec(magnitudes[:, np.newaxis] * directions).as_matrix()


def test_od():
    # update_brahe_data_files()
    config = load_config()

    # set up landmark bearing sensor and orbit determination objects
    # landmark_bearing_sensor = RandomLandmarkBearingSensor(config)
    landmark_bearing_sensor = SimulatedMLLandmarkBearingSensor(config)
    od = OrbitDetermination(dt=1 / config["solver"]["world_update_rate"])

    # set up initial state
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] * config["solver"]["world_update_rate"]))
    states = np.zeros((N, 6))
    # pick a latitude and longitude that results in the satellite passing over the contiguous US in its first few orbits
    states[0, :] = get_sso_orbit_state(starting_epoch, 0, -73, 600e3, northwards=True)
    epoch = starting_epoch

    # set up arrays to store measurements
    times = np.array([], dtype=int)
    Rs_body_to_eci = np.zeros(shape=(0, 3, 3))
    bearing_unit_vectors = np.zeros(shape=(0, 3))
    landmarks = np.zeros(shape=(0, 3))

    def take_measurement(t_idx: int) -> None:
        """
        Take a set of measurements at the given time index.
        Reads from the states and landmark_bearing_sensor variables in the outer scope.
        Appends to the times, Rs_body_to_eci, bearing_unit_vectors, and landmarks arrays in the outer scope.

        :param t_idx: The time index at which to take the measurements.
        """
        position = states[t_idx, :3]
        R_body_to_eci = get_nadir_rotation(position)

        measurement_bearing_unit_vectors, measurement_landmarks = (
            landmark_bearing_sensor.take_measurement(epoch, position, R_body_to_eci)
        )
        measurement_count = measurement_bearing_unit_vectors.shape[0]
        assert measurement_landmarks.shape[0] == measurement_count

        nonlocal times, Rs_body_to_eci, bearing_unit_vectors, landmarks
        times = np.concatenate((times, np.repeat(t_idx, measurement_count)))
        Rs_body_to_eci = np.concatenate(
            (Rs_body_to_eci, np.tile(R_body_to_eci, (measurement_count, 1, 1))), axis=0
        )
        bearing_unit_vectors = np.concatenate(
            (bearing_unit_vectors, measurement_bearing_unit_vectors), axis=0
        )
        landmarks = np.concatenate((landmarks, measurement_landmarks), axis=0)
        print(f"Total measurements so far: {len(times)}")
        print(f"Completion: {100 * t_idx / N:.2f}%")

    for t in range(0, N - 1):
        states[t + 1, :] = f(states[t, :], od.dt)

        if t % 5 == 0 and is_over_daytime(
            epoch, states[t, :3]
        ):  # take a set of measurements every 5 minutes
            take_measurement(t)

        epoch = increment_epoch(epoch, 1 / config["solver"]["world_update_rate"])

    if len(times) == 0:
        raise ValueError("No measurements taken")
    print(f"Total measurements: {len(times)}")

    if type(landmark_bearing_sensor) == SimulatedMLLandmarkBearingSensor:
        # save measurements to pickle file
        with open(f"measurements-{time()}.pkl", "wb") as file:
            pickle.dump(
                {
                    "times": times,
                    "states": states,
                    "Rs_body_to_eci": Rs_body_to_eci,
                    "bearing_unit_vectors": bearing_unit_vectors,
                    "landmarks": landmarks,
                },
                file,
            )

    # for i, attitude_noise in enumerate(attitude_noises):
    #     so3_noise_matrices = get_SO3_noise_matrices(len(times), np.deg2rad(attitude_noise))
    #     bearing_unit_vectors = np.einsum("ijk,ik->ij", so3_noise_matrices, bearing_unit_vectors)
    #
    #     start_time = perf_counter()
    #     estimated_states = od.fit_orbit(times, landmarks, bearing_unit_vectors, Rs_body_to_eci, N)
    #     print(f"Elapsed time: {perf_counter() - start_time:.2f} s")
    #
    #     position_errors = np.linalg.norm(states[:, :3] - estimated_states[:, :3], axis=1)
    #     rms_position_error = np.sqrt(np.mean(position_errors ** 2))
    #     print(f"Attitude SO(3) Noise Variance: {attitude_noise}")
    #     print(f"RMS position error: {rms_position_error}")
    #     print(f"Completion Percentage: {100 * (i + 1) / len(attitude_noises):.2f}%")
    #
    #     rms_position_errors[i] = rms_position_error
    #
    # print(f"{rms_position_errors=}")
    # plt.figure()
    # plt.xlabel("Attitude SO(3) Noise Variance [deg]")
    # plt.ylabel("OD RMS Position Error [km]")
    # plt.title("Effect of Attitude Noise on Orbit Determination")
    # plt.scatter(np.rad2deg(attitude_noises), rms_position_errors / 1e3)
    # plt.plot([0, 10], [50, 50], linestyle="--", color="r")
    # plt.show()

    start_time = perf_counter()
    estimated_states = od.fit_orbit(times, landmarks, bearing_unit_vectors, Rs_body_to_eci, N)
    print(f"Elapsed time: {perf_counter() - start_time:.2f} s")

    position_errors = np.linalg.norm(states[:, :3] - estimated_states[:, :3], axis=1)
    rms_position_error = np.sqrt(np.mean(position_errors**2))
    print(f"RMS position error: {rms_position_error}")

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(-1.5 * R_EARTH, 1.5 * R_EARTH)
    # ax.set_ylim(-1.5 * R_EARTH, 1.5 * R_EARTH)
    # ax.set_zlim(-1.5 * R_EARTH, 1.5 * R_EARTH)
    #
    # ax.plot(states[:, 0], states[:, 1], states[:, 2], label="True orbit")
    # ax.plot(estimated_states[:, 0], estimated_states[:, 1], estimated_states[:, 2], label="Estimated orbit")
    # ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], label="Landmarks")
    #
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")
    # ax.legend()
    # plt.show()


if __name__ == "__main__":
    np.random.seed(69420)
    test_od()

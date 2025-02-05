"""
Testing the EKF class.
"""

import csv
import math
import os
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brahe.constants import e_EARTH
from brahe.constants import GM_EARTH
from brahe.constants import R_EARTH
import numpy as np
from scipy.spatial.transform import Rotation

from orbit_determination.ekf import EKF
from dynamics.orbital_dynamics import f
from dynamics.orbital_dynamics import f_jac
import quaternion
from sensors.imu import IMU
from sensors.imu import IMU
from sensors.sensor import SensorNoiseParams
from sensors.bias import BiasParams
from sensors.imu import IMUNoiseParams
from utils.earth_utils import is_visible_on_ellipse


def latlon2ecef(landmarks: list) -> np.ndarray:
    """
    Convert latitude, longitude, and altitude coordinates to Earth-Centered, Earth-Fixed (ECEF) coordinates.

    Args:
        landmarks (list of tuples): A list of tuples where each tuple contains:
            - landmark[0] (any): An identifier for the landmark.
            - landmark[1] (float): Latitude in radians.
            - landmark[2] (float): Longitude in radians.
            - landmark[3] (float): Altitude in kilometers.
    Returns:
        numpy.ndarray: A 2D array where each row corresponds to a landmark and contains:
            - landmark[0] (any): The identifier for the landmark.
            - X (float): The ECEF X coordinate in kilometers.
            - Y (float): The ECEF Y coordinate in kilometers.
            - Z (float): The ECEF Z coordinate in kilometers.
    """
    ecef = []

    R_EARTH_EQ = R_EARTH
    R_EARTH_POL = R_EARTH * (1 - e_EARTH**2) ** 0.5

    # helper function
    def N(a, b, lat):
        return a**2 / np.sqrt(a**2 * np.cos(lat) ** 2 + b**2 * np.sin(lat) ** 2)

    for mark in landmarks:
        lat_deg = float(mark[1])
        lon_deg = float(mark[2])
        h = float(mark[3])

        # Convert degrees to radians
        lat_rad = np.deg2rad(lat_deg)
        lon_rad = np.deg2rad(lon_deg)

        N_val = N(R_EARTH_EQ, R_EARTH_POL, lat_rad)
        X = (N_val + h) * np.cos(lat_rad) * np.cos(lon_rad)
        Y = (N_val + h) * np.cos(lat_rad) * np.sin(lon_rad)
        Z = (N_val * (1 - e_EARTH**2) + h) * np.sin(lat_rad)

        ecef.append([mark[0], X, Y, Z])

    return np.array(ecef)


def import_landmarks() -> List["landmark"]:
    """
    Imports landmark coordinates from a CSV file, converts them to ECEF coordinates,
    and initializes landmark objects.

    Returns:
        List[landmark]: A list of landmark objects with their names and ECEF coordinates.

    Raises:
        FileNotFoundError: If the landmark coordinates file is not found.
        ValueError: If there is an error reading the landmark coordinates.
    """

    landmarks = []
    try:
        with open(
            "orbit_determination/landmark_coordinates.csv", newline="", encoding="utf-8"
        ) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                landmarks.append(np.array([row[0], row[1], row[2], row[3]]))

        landmarks_ecef = latlon2ecef(landmarks)
        landmark_objects = []

        for landmark_obj in landmarks_ecef:
            landmark_objects.append(
                landmark(
                    x=float(landmark_obj[1]),
                    y=float(landmark_obj[2]),
                    z=float(landmark_obj[3]),
                    name=(landmark_obj[0]),
                )
            )

        return landmark_objects
    except FileNotFoundError as exc:
        raise FileNotFoundError("Landmark coordinates file not found.") from exc
    except ValueError as exc:
        raise ValueError(f"Error reading landmark coordinates: {exc}") from exc


class landmark:  # Class for the landmark object.
    def __init__(self, x: float, y: float, z: float, name: str) -> None:
        self.pos = np.array([x, y, z])
        self.name = name


def visible_landmarks_list(position, landmark_objects) -> List[landmark]:
    """
    Get a list of landmarks that are visible from a given position.
    """
    curr_visible_landmarks = []
    for landmark in landmark_objects:
        if is_visible_on_ellipse(position, landmark.pos):
            curr_visible_landmarks.append(landmark)

    return curr_visible_landmarks


def measure_z_landmark(curr_pos, curr_quat, curr_visible_landmarks) -> np.ndarray:
    """
    Measure the landmark bearing from the satellite to the landmarks and add noise
    curr_pos represents the ground truth position of the satellite.
    TODO: Need to make noise multiplicative rather than additive for measurements of type bearing
    """
    z_l = np.zeros((len(curr_visible_landmarks) * 3))
    for i, landmark in enumerate(curr_visible_landmarks):
        noise = np.random.normal(loc=0, scale=math.sqrt(0), size=(3))
        vec = curr_pos - landmark.pos + noise
        vec = vec / np.linalg.norm(vec)
        # body_R_eci = Rotation.from_quat(curr_quat, scalar_first=True).as_matrix()
        # body_vec = body_R_eci @ vec
        z_l[i * 3 : i * 3 + 3] = vec
    return z_l


def run_simulation(landmark_test_objects):
    dt = 10
    timesteps = 1000

    ground_truth = np.zeros((timesteps + 1, 6))
    ground_truth_quat = np.zeros((timesteps + 1, 4))

    orbit_height = 590  # in km
    init_state = np.array([(R_EARTH + orbit_height * 1000)*1, 0, 0, 0, 10, 7500])
    init_quat = np.array([1, 0, 0, 0])

    ground_truth[0, :] = init_state
    ground_truth_quat[0, :] = init_quat

    rot = np.array([0, 0, np.pi / 2])

    for i in range(timesteps):
        # Using RK4 for state propagation of position and velocity as it's already implemented
        ground_truth[i + 1, :] = f(ground_truth[i, :], dt)
        # Using Euler for quaternion state propagation as RK4 doesn't yet exist. Store as float array
        tmp_quat = quaternion.as_quat_array(ground_truth_quat[i, :])
        ground_truth_quat[i + 1, :] = quaternion.as_float_array(
            tmp_quat * quaternion.from_rotation_vector(0.5 * dt * rot)
        )
        # TODO: IMU runs at a higher rate than the rest of the system so probably better to introduce a separate dt for it

    # Initialize the IMU
    bias_params = BiasParams.get_random_params([0, 0], [1e-4, 1e-3])
    sensor_noise_params_accel_x = SensorNoiseParams(bias_params, 5e-4, 5e-4)
    sensor_noise_params_accel_y = SensorNoiseParams(bias_params, 5e-4, 5e-4)
    sensor_noise_params_accel_z = SensorNoiseParams(bias_params, 5e-4, 5e-4)
    sensor_noise_params_accel = [
        sensor_noise_params_accel_x,
        sensor_noise_params_accel_y,
        sensor_noise_params_accel_z,
    ]

    sensor_noise_params_gyro_x = SensorNoiseParams(bias_params, 5e-4, 5e-4)
    sensor_noise_params_gyro_y = SensorNoiseParams(bias_params, 5e-4, 5e-4)
    sensor_noise_params_gyro_z = SensorNoiseParams(bias_params, 5e-4, 5e-4)
    sensor_noise_params_gyro = [
        sensor_noise_params_gyro_x,
        sensor_noise_params_gyro_y,
        sensor_noise_params_gyro_z,
    ]

    imu_noise_params = IMUNoiseParams(
        gyro_params=sensor_noise_params_gyro, accel_params=sensor_noise_params_accel
    )
    imu = IMU(dt, imu_noise_params)

    # Initialize the EKF
    ekf = EKF(
        r=init_state[0:3] + np.random.normal(0, 100, 3),
        v=init_state[3:6] + np.random.normal(0, 3, 3),
        q=quaternion.as_quat_array(init_quat),
        P=np.eye(6) * 10,
        Q=np.eye(6) * 1e-12,
        R=np.zeros((3, 3)),
        dt=dt,
    )

    # Run the simulation
    for i in range(timesteps):
        print("timestep", i)
        # Get the IMU measurements
        imu_update_truth = quaternion.as_quat_array(ground_truth_quat[i + 1, :])  # Stored as float array
        imu_update_truth = quaternion.as_rotation_vector(imu_update_truth)
        gyro_meas, _ = imu.update(imu_update_truth, [0, 0, 0])  # Adds noise to the IMU measurements

        # Run EKF prediction step
        # print(quaternion.from_rotation_vector(gyro_meas))
        ekf.predict(u=gyro_meas)

        # Get the measurements
        # Take the satellite position in ECEF coordinates for the measurement
        # Ignore the actual attitude of the satellite in the determination of visible landmarks
        visible_landmarks = visible_landmarks_list(ground_truth[i + 1, :3], landmark_test_objects)
        z_landmark = measure_z_landmark(
            ground_truth[i + 1, :3], ground_truth_quat[i + 1, :], visible_landmarks
        )

        landmark_list = []
        for _, landmark in enumerate(visible_landmarks):
            landmark_list.append(landmark.pos)
        landmark_list = np.array(landmark_list)
        z = (z_landmark, landmark_list)
        
        # Run the EKF post-update step
        if i == timesteps - 1:
            print("debug")

        ekf.measurement(z)
        if landmark_list.shape[0] > 0:
            print("landmark seen")
        print("error", ekf.r_m - ground_truth[i + 1, :3])


if __name__ == "__main__":
    # Initialize the landmarks
    landmark_test_objects = import_landmarks()

    # Initalize the EKF
    # Run state propagation for the satellite based on ICs
    run_simulation(landmark_test_objects)

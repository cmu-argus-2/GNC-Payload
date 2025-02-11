"""
Testing the EKF class.
"""

import os
import sys
from time import time
from typing import Any
from typing import List
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import brahe
from brahe.constants import GM_EARTH
from brahe.constants import R_EARTH
from brahe.epoch import Epoch
import numpy as np
import pickle
import yaml

from dynamics.orbital_dynamics import f
from dynamics.orbital_dynamics import f_jac
from orbit_determination.ekf import EKF
from orbit_determination.landmark_bearing_sensors import SimulatedMLLandmarkBearingSensor
from orbit_determination.landmark_bearing_sensors import GroundTruthLandmarkBearingSensor
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
import quaternion
from sensors.imu import IMU
from sensors.sensor import SensorNoiseParams
from sensors.bias import BiasParams
from sensors.imu import IMUNoiseParams
from utils.orbit_utils import get_sso_orbit_state
from utils.orbit_utils import is_over_daytime


def load_config() -> dict[str, Any]:
    """
    Load the configuration file and modify it for the purposes of this test.

    :return: The modified configuration file as a dictionary.
    """
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # TODO: move this into the config file itself
    # decrease world update rate since we only care about position dynamics
    config["solver"]["world_update_rate"] = 1 / 60  # Hz
    config["mission"]["duration"] = 3 * 90 * 60  # s, roughly 1 orbit

    return config


def imu_init(dt):
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

    return imu


def run_simulation():

    config = load_config()

    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))  # number of time steps in the simulation

    landmark_bearing_sensor = GroundTruthLandmarkBearingSensor(config)
    data_manager = ODSimulationDataManager(starting_epoch, dt)

    initial_state = get_sso_orbit_state(starting_epoch, 0, -73, 600e3, northwards=True)
    init_rot = quaternion.as_rotation_matrix(quaternion.as_quat_array(np.array([1, 0, 0, 0])))

    data_manager.push_next_state(
        np.expand_dims(initial_state, axis=0), np.expand_dims(init_rot, axis=0)
    )

    # Initialize IMU and EKF
    imu = imu_init(dt)
    ekf = EKF(
        r=initial_state[0:3] + np.random.normal(0, 0, 3),  # TODO: Adjust and tune noise and error init
        v=initial_state[3:6] + np.random.normal(0, 0, 3),
        q=quaternion.from_rotation_matrix(init_rot),
        P=np.eye(6) * 1,
        Q=np.eye(6) * 1e-12,
        R=np.zeros((3, 3)),
        dt=dt,
    )

    # Fix a constant rotation velocity for the test.
    rot = np.array([0, 0, np.pi / 2])

    error = []

    for t in range(0, N - 1):
        # take a set of measurements every minute

        next_state = f(data_manager.latest_state, dt)
        curr_quat = quaternion.from_rotation_matrix(data_manager.latest_attitude)
        next_quat = curr_quat * quaternion.from_rotation_vector(0.5 * dt * rot)
        data_manager.push_next_state(
            np.expand_dims(next_state, axis=0),
            np.expand_dims(quaternion.as_rotation_matrix(next_quat), axis=0),
        )
        
        if t % 1 == 0:
            data_manager.take_measurement(landmark_bearing_sensor)
            print(f"Total measurements so far: {data_manager.measurement_count}")
            print(f"Completion: {100 * t / N:.2f}%")

        # EKF prediction step
        gyro_meas = np.zeros((3))
        # gyro_meas, _ = imu.update(quaternion.as_rotation_vector(next_quat), [0, 0, 0])
        ekf.predict(u=gyro_meas)
        z = (data_manager.curr_bearing_unit_vectors, data_manager.curr_landmarks)
        ekf.measurement(z)
        
        error.append(ekf.r_m - next_state[0:3])

    if type(landmark_bearing_sensor) == SimulatedMLLandmarkBearingSensor:
        # save measurements to pickle file
        with open(f"od-simulation-data-{time()}.pkl", "wb") as file:
            pickle.dump(data_manager, file)

    plt.plot(error)
    plt.show()
    # TODO: IMU runs at a higher rate than the rest of the system so probably better to introduce a separate dt for it

    # # Run the simulation
    # for i in range(timesteps):

    #     # Get the measurements
    #     # Take the satellite position in ECEF coordinates for the measurement
    #     # Ignore the actual attitude of the satellite in the determination of visible landmarks
    #     # visible_landmarks = visible_landmarks_list(ground_truth[i + 1, :3], landmark_test_objects)
    #     # z_landmark = measure_z_landmark(
    #     #     ground_truth[i + 1, :3], ground_truth_quat[i + 1, :], visible_landmarks
    #     # )

    #     landmark_list = []
    #     for _, landmark in enumerate(visible_landmarks):
    #         landmark_list.append(landmark.pos)
    #     landmark_list = np.array(landmark_list)
    #     z = (z_landmark, landmark_list)

    #     # Run the EKF post-update step
    #     if i == timesteps - 1:
    #         print("debug")

    #     ekf.measurement(z)
    #     if landmark_list.shape[0] > 0:
    #         print("landmark seen")
    #     print("error", ekf.r_m - ground_truth[i + 1, :3])


if __name__ == "__main__":
    # Run state propagation for the satellite based on ICs
    run_simulation()

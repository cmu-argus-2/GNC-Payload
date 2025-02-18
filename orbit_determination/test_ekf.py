"""
Testing the EKF class.
"""

import pickle
from time import time

import brahe
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from brahe.epoch import Epoch

from dynamics.orbital_dynamics import f
from orbit_determination.ekf import EKF
from orbit_determination.landmark_bearing_sensors import (
    GroundTruthLandmarkBearingSensor,
    SimulatedMLLandmarkBearingSensor,
)
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from sensors.bias import BiasParams
from sensors.imu import IMU, IMUNoiseParams
from sensors.sensor import SensorNoiseParams
from utils.config_utils import load_config
from utils.orbit_utils import get_sso_orbit_state  # , is_over_daytime


def imu_init(dt: float) -> IMU:
    """
    Initializes the IMU.

    :param dt: The time step for the simulation.

    :return: The initialized IMU.
    """
    # Initialize the IMU
    bias_params = BiasParams.get_random_params([1e-6, 1e-5], [1e-6, 1e-5])
    sensor_noise_params_accel_x = SensorNoiseParams(bias_params, 5e-10, 5e-8)
    sensor_noise_params_accel_y = SensorNoiseParams(bias_params, 5e-10, 5e-8)
    sensor_noise_params_accel_z = SensorNoiseParams(bias_params, 5e-10, 5e-8)
    sensor_noise_params_accel = [
        sensor_noise_params_accel_x,
        sensor_noise_params_accel_y,
        sensor_noise_params_accel_z,
    ]

    sensor_noise_params_gyro_x = SensorNoiseParams(bias_params, 5e-10, 5e-9)
    sensor_noise_params_gyro_y = SensorNoiseParams(bias_params, 5e-10, 5e-9)
    sensor_noise_params_gyro_z = SensorNoiseParams(bias_params, 5e-10, 5e-9)
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


def run_simulation() -> None:
    """
    Run the simulation.

    :return: None
    """

    config = load_config()

    config["solver"]["world_update_rate"] = 1 / 2  # Hz
    config["mission"]["duration"] = 3 * 90 * 30  # s, roughly 1 orbit

    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))  # number of time steps in the simulation

    landmark_bearing_sensor = GroundTruthLandmarkBearingSensor(config)
    data_manager = ODSimulationDataManager(starting_epoch, dt)

    initial_state = get_sso_orbit_state(starting_epoch, 0, -73, 600e3, northwards=True)
    init_rot = np.eye(3)

    data_manager.push_next_state(initial_state, init_rot)

    # Set the number of update iterations for the IEKF
    num_iter = 5

    # Fix a constant rotation velocity for the test.
    rot = np.array([0, 0, np.pi / 3])

    # Initialize IMU and EKF
    imu = imu_init(dt)
    gyro_bias = imu.get_bias()[0]
    ekf = EKF(
        # TODO: Apply initial error to quaternion initialization
        
        r=initial_state[0:3] + np.random.normal(0, 100, 3),
        v=initial_state[3:6] + np.random.normal(0, 100, 3),
        q=quaternion.as_float_array(quaternion.from_rotation_matrix(init_rot)),
        P=np.eye(12) * 30,
        Q=np.eye(12) * 1e-12,
        R_vec=np.zeros((3, 3)),
        dt=dt,
        config=config,
        w_b=gyro_bias + np.random.normal(0, 1e-5, 3),
    )

    error = []
    for t in range(0, N - 1):
        # take a set of measurements every minute
        w = rot
        x = data_manager.latest_state
        q = data_manager.latest_attitude

        bias = imu.get_bias()[0]
        # x = np.concatenate([x, quaternion.as_float_array(quaternion.from_rotation_matrix(q)), w])

        next_state = f(x, dt)
        next_quat = quaternion.from_rotation_matrix(q) * quaternion.from_rotation_vector(
            (w - bias) * dt * 0.5
        )

        data_manager.push_next_state(next_state, quaternion.as_rotation_matrix(next_quat))

        # gyro_meas = np.zeros((3))  # TEMPORARY

        gyro_meas, _ = imu.update(w, np.zeros((3)))
        ekf.predict(u=gyro_meas)

        if t % 4 == 0:
            data_manager.take_measurement(landmark_bearing_sensor)
            print(f"Total measurements so far: {data_manager.measurement_count}")
            print(f"Completion: {100 * t / N:.2f}%")

            # EKF prediction step
            z = data_manager.latest_measurements
            if z[0].shape[0] > 0:
                ekf.measurement(z, data_manager, num_iter)
            else:
                ekf.no_measurement()
        else:
            ekf.no_measurement()

        error.append(ekf.r_m - next_state[0:3])

    if isinstance(landmark_bearing_sensor, SimulatedMLLandmarkBearingSensor):
        # save measurements to pickle file
        with open(f"od-simulation-data-{time()}.pkl", "wb") as file:
            pickle.dump(data_manager, file)

    plt.plot(error)
    plt.legend(["x", "y", "z"])
    plt.xlabel("Time step")
    plt.ylabel("Position error [m]")
    plt.title("EKF Position Error")
    plt.show()
    # TODO: IMU runs at a higher rate than the rest of the system so probably better to introduce a separate dt for it


if __name__ == "__main__":
    # Run state propagation for the satellite based on ICs
    run_simulation()

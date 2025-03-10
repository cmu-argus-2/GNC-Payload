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

from dynamics.orbital_dynamics import Dynamics
from orbit_determination.ekf import EKF
from orbit_determination.landmark_bearing_sensors import (
    GroundTruthLandmarkBearingSensor,
    SimulatedMLLandmarkBearingSensor,
)
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from sensors.bias import BiasParams
from sensors.camera_model import CameraModelManager
from sensors.imu import IMU, IMUNoiseParams
from sensors.sensor import SensorNoiseParams
from utils.brahe_utils import load_brahe_data_files_if_needed
from utils.config_utils import load_config
from utils.orbit_utils import get_sso_orbit_state  # , is_over_daytime

# pylint: disable=too-many-locals


def imu_init(dt: float) -> IMU:
    """
    Initializes the IMU.

    :param dt: The time step for the simulation.

    :return: The initialized IMU.
    """
    # Initialize the IMU
    # bias params are min max range of bias and sigma_w
    # [units] and [(units/s)/sqrt(Hz)]
    bias_params = BiasParams.get_random_params([0, 0], [0, 0])
    # sigma_v [units/sqrt(Hz)] & scale_factor_error [-]
    sensor_noise_params_accel_x = SensorNoiseParams(bias_params, 5e-10, 5e-9)
    sensor_noise_params_accel_y = SensorNoiseParams(bias_params, 5e-10, 5e-9)
    sensor_noise_params_accel_z = SensorNoiseParams(bias_params, 5e-10, 5e-9)
    sensor_noise_params_accel = [
        sensor_noise_params_accel_x,
        sensor_noise_params_accel_y,
        sensor_noise_params_accel_z,
    ]
    # sigma_v [units/sqrt(Hz)] & scale_factor_error [-]
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
    # Set the world update rate and mission duration to a rate that is workable for testing
    config["solver"]["world_update_rate"] = 6  # Hz
    config["mission"]["duration"] = 3 * 90 * 40  # s

    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))  # number of time steps in the simulation

    landmark_bearing_sensor = GroundTruthLandmarkBearingSensor()
    camera_model_manager = CameraModelManager()
    data_manager = ODSimulationDataManager(starting_epoch, dt)

    initial_state = get_sso_orbit_state(starting_epoch, 0, -73, 600e3, northwards=True)
    init_rot = np.eye(3)

    data_manager.push_next_state(initial_state, init_rot)

    # Set the number of update iterations for the IEKF
    num_iter = 4

    # Fix a constant rotation velocity for the test.
    rot = np.array([0, 0, np.pi / 18])

    # Prep Q matrix for the EKF. Unmodelled attitude has larger uncertainty
    Q = np.eye(12) * 1e-12
    Q[6:9, 6:9] = np.eye(3) * 1e-5

    # Set up dynamics instance for ground truth and EKF
    ground_truth_dynamics = Dynamics(
        config=config,
        data_manager=data_manager,
        use_drag=True,
        use_j2=True,
        use_unmodelled_a=False,
    )
    ekf_dynamics = Dynamics(
        config=config,
        data_manager=data_manager,
        use_drag=False,
        use_j2=False,
        use_unmodelled_a=True,
    )

    # Initialize IMU and EKF
    imu = imu_init(dt)
    ekf = EKF(
        # TODO: Apply initial error to quaternion initialization
        # error ranges are in meters and m/s
        r=initial_state[0:3] + np.random.normal(0, 5000, 3),
        v=initial_state[3:6] + np.random.normal(0, 10, 3),
        ua=np.random.normal(0, 1e-5, 3),
        q=quaternion.as_float_array(quaternion.from_rotation_matrix(init_rot)),
        P=np.eye(12) * 5,
        Q=Q,
        dt=dt,
        config=config,
        data_manager=data_manager,
        ekf_dynamics=ekf_dynamics,
    )

    error = []
    vel_error = []
    ua_error = []
    cov_trace = []

    for t in range(0, N - 1):
        # take a set of measurements every minute
        x = data_manager.latest_state
        x = np.concatenate([x, ekf.ua])
        q = data_manager.latest_attitude

        # Apply noise to x, y to generate angular wobble around the primary rotation axis z
        x_y_wobble = np.random.normal(0, 5e-2, 2)
        w = rot + np.concatenate([x_y_wobble, np.zeros((1))])

        next_state = ground_truth_dynamics.perturbed_f(x=x[0:6], dt=dt)
        next_quat = quaternion.from_rotation_matrix(q) * quaternion.from_rotation_vector(w * dt)

        data_manager.push_next_state(next_state[0:6], quaternion.as_rotation_matrix(next_quat))

        gyro_meas, _ = imu.update(w, np.zeros((3)))
        ekf.predict(u=gyro_meas)

        if t % 150 == 0:
            for camera_name in CameraModelManager.CAMERA_NAMES:
                data_manager.take_measurement(
                    landmark_bearing_sensor, camera_model_manager[camera_name]
                )
            print(f"Total measurements so far: {data_manager.measurement_count}")
            print(f"Completion: {100 * t / N:.2f}%")

            # EKF prediction step
            measurement_camera_names, *z = data_manager.latest_measurements

            if z[0].shape[0] > 0:
                ekf.measurement(z, camera_model_manager, measurement_camera_names, num_iter)
            else:
                ekf.no_measurement()
                print("No measurements made in measurement step")
        else:
            ekf.no_measurement()

        error.append(ekf.r_m - next_state[0:3])
        vel_error.append(ekf.v_m - next_state[3:6])
        ua_error.append(ekf.ua)
        cov_trace.append(np.trace(ekf.P_m))

    if isinstance(landmark_bearing_sensor, SimulatedMLLandmarkBearingSensor):
        # save measurements to pickle file
        with open(f"od-simulation-data-{time()}.pkl", "wb") as file:
            pickle.dump(data_manager, file)

    plt.plot(error)
    plt.legend(["x", "y", "z"])
    plt.xlabel("Time step")
    plt.ylabel("Position error [m]")
    plt.title("EKF Position Error")

    plt.figure()

    plt.plot(vel_error)
    plt.legend(["x", "y", "z"])
    plt.xlabel("Time step")
    plt.ylabel("Velocity error [m/s]")
    plt.title("EKF Velocity Error")

    plt.figure()

    plt.plot(ua_error)
    plt.legend(["x", "y", "z"])
    plt.xlabel("Time step")
    plt.ylabel("Unmodelled acc error [m/s^2]")
    plt.title("EKF Unmodelled Acceleration Error")

    plt.figure()

    plt.plot(cov_trace)
    plt.xlabel("Time step")
    plt.ylabel("Covariance trace")
    plt.title("EKF Covariance Trace")

    plt.show()
    # TODO: IMU runs at a higher rate than the rest of the system so probably better to introduce a separate dt for it


if __name__ == "__main__":
    # Run state propagation for the satellite based on ICs
    load_brahe_data_files_if_needed()
    run_simulation()

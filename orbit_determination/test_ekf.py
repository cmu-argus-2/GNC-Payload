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

from dynamics.ekf_dynamics import EKFDynamics
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
from utils.orbit_utils import get_sso_orbit_state, is_over_daytime

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
    bias_params_x = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
    bias_params_y = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
    bias_params_z = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
    # bias_params = BiasParams.get_random_params([0, 0], [0, 0])
    # sigma_v [units/sqrt(Hz)] & scale_factor_error [-]
    sensor_noise_params_accel_x = SensorNoiseParams.get_random_params(
        bias_params_x, [0, 0.0], [0, 0.0]
    )
    sensor_noise_params_accel_y = SensorNoiseParams.get_random_params(
        bias_params_y, [0, 0.0], [0, 0.0]
    )
    sensor_noise_params_accel_z = SensorNoiseParams.get_random_params(
        bias_params_z, [0, 0.0], [0, 0.0]
    )
    sensor_noise_params_accel = [
        sensor_noise_params_accel_x,
        sensor_noise_params_accel_y,
        sensor_noise_params_accel_z,
    ]
    # sigma_v [units/sqrt(Hz)] & scale_factor_error [-]
    sensor_noise_params_gyro_x = SensorNoiseParams.get_random_params(
        bias_params_x, [1e-6, 1e-5], [0, 0.01]
    )
    sensor_noise_params_gyro_y = SensorNoiseParams.get_random_params(
        bias_params_y, [1e-6, 1e-5], [0, 0.01]
    )
    sensor_noise_params_gyro_z = SensorNoiseParams.get_random_params(
        bias_params_z, [1e-6, 1e-5], [0, 0.01]
    )
    sensor_noise_params_gyro = [
        sensor_noise_params_gyro_x,
        sensor_noise_params_gyro_y,
        sensor_noise_params_gyro_z,
    ]

    imu_noise_params = IMUNoiseParams(
        gyro_params=sensor_noise_params_gyro, accel_params=sensor_noise_params_accel
    )
    imu = IMU(
        dt=dt,
        IMU_noise_params=imu_noise_params,
        misalignment_range=[-0.01, 0.01],
    )

    return imu


def run_simulation() -> None:
    """
    Run the simulation.

    :return: None
    """

    config = load_config()
    # Set the world update rate and mission duration to a rate that is workable for testing
    config["solver"]["world_update_rate"] = 2  # Hz
    config["mission"]["duration"] = 3 * 90 * 45  # s

    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))  # number of time steps in the simulation

    landmark_bearing_sensor = GroundTruthLandmarkBearingSensor()
    camera_model_manager = CameraModelManager()
    data_manager = ODSimulationDataManager(starting_epoch, dt)

    initial_state = get_sso_orbit_state(starting_epoch, 0, -73, 600e3, northwards=True)
    # initial_state = initial_state / 1e3  # Convert from m to km and m/s to km/s
    # Set the initial rotation matrix to identity
    init_rot = np.eye(3)

    data_manager.push_next_state(initial_state, init_rot)

    # Apply error to init_rot and ensure orthonormality
    noisy_rot = init_rot + np.random.normal(0, 1e-2, (3, 3))
    noisy_rot = noisy_rot @ np.linalg.inv(np.linalg.cholesky(noisy_rot.T @ noisy_rot))

    # Assert orthonormality
    assert np.allclose(noisy_rot @ noisy_rot.T, np.eye(3), atol=1e-3) and np.isclose(
        np.linalg.det(noisy_rot), 1
    ), "Rotation matrix is not a proper rotation matrix"

    # Set the number of update iterations for the IEKF
    num_iter = 5

    # Set up scaling parameter for the unmodelled acceleration
    ua_scale = 10

    # Set up scaling parameter for gyro bias
    gyro_bias_scale = 2

    # Fix a constant rotation velocity for the test.
    rot = np.array([0, 0, np.pi / 18])

    # Prep Q matrix for the EKF.
    Q = np.eye(16) * 1e-12
    # Unmodelled acceleration has larger uncertainty
    Q[6:9, 6:9] = np.eye(3) * 1e-9
    # # Bias uncertainty also larger
    Q[13:16, 13:16] = np.eye(3) * 1e-9

    P = np.eye(16)
    P[0:3, 0:3] *= 5
    P[3:6, 3:6] *= 5
    P[6:9, 6:9] *= 1e-4
    P[9:10, 9:10] *= 1e-4
    P[10:13, 10:13] *= 1e-4
    P[13:16, 13:16] *= 1e-4

    # Set up dynamics instance for ground truth and EKF
    ground_truth_dynamics = Dynamics(
        config=config,
        use_drag=True,
        use_j2=True,
        use_j34=True,
        use_sun_grav=True,
        use_moon_grav=True,
    )
    ekf_dynamics = EKFDynamics(
        config=config,
        use_drag=False,
        use_j2=False,
        use_j34=False,
        use_sun_grav=False,
        use_moon_grav=False,
        use_unmodelled_a=True,
        use_drag_scalar=True,
        ua_scale=ua_scale,
    )

    # Initialize IMU and EKF
    imu = imu_init(dt)
    gyro_bias = (imu.get_bias()[0] + np.random.normal(0, 5e-5, 3)) * gyro_bias_scale
    ekf = EKF(
        # error ranges are in meters and m/s
        r=initial_state[0:3] + np.random.normal(0, 5000, 3),
        v=initial_state[3:6] + np.random.normal(0, 10, 3),
        ua=np.random.normal(0, 1e-5, 3) * ua_scale,
        q=quaternion.as_float_array(quaternion.from_rotation_matrix(noisy_rot)),
        P=P,
        Q=Q,
        dt=dt,
        config=config,
        ekf_dynamics=ekf_dynamics,
        w_b=gyro_bias,
        gyro_bias_scale=gyro_bias_scale,
    )

    # Store errors for plotting
    error = []
    vel_error = []
    ua_error = []
    cov_trace = []
    gyro_bias_error = []
    actual_bias = []
    drag_estimate = []
    sigma_high = []
    sigma_low = []

    for t in range(0, N - 1):
        # take a set of measurements every minute
        x = data_manager.latest_state
        x = np.concatenate([x, ekf.ua])
        q = data_manager.latest_attitude

        # Apply noise to x, y to generate angular wobble around the primary rotation axis z
        # One rotation every 10 seconds to model a relatively slow wobble
        w = rot + 0.05 * np.array(
            [np.cos(2 * np.pi * t / (10 / dt)), np.sin(2 * np.pi * t / (10 / dt)), 0]
        )

        # Get a gyro measurement to use in the EKF and the current gyro bias for the ground truth
        gyro_meas, _ = imu.update(w, np.zeros((3)))
        imu_gyro_bias = imu.get_bias()[0]

        next_state = ground_truth_dynamics.perturbed_f(
            x=x[0:6], dt=dt, epoch=data_manager.latest_epoch
        )
        next_quat = quaternion.from_rotation_matrix(q) * quaternion.from_rotation_vector(w * dt)

        data_manager.push_next_state(next_state[0:6], quaternion.as_rotation_matrix(next_quat))

        ekf.predict(u=gyro_meas, epoch=data_manager.latest_epoch)

        if t % 120 == 0 and is_over_daytime(
            data_manager.latest_epoch, data_manager.latest_state[:3]
        ):
            for camera_name in CameraModelManager.CAMERA_NAMES:
                data_manager.take_measurement(
                    landmark_bearing_sensor, camera_model_manager[camera_name]
                )
            print(f"Total measurements so far: {data_manager.measurement_count}")
            print(f"Completion: {100 * t / N:.2f}%")
            print(f"State position: {next_state[0:3]}")

            # EKF prediction step
            measurement_camera_names, *z = data_manager.latest_measurements

            if z[0].shape[0] > 0:
                ekf.measurement(
                    z=z,
                    camera_model_manager=camera_model_manager,
                    measurement_camera_names=measurement_camera_names,
                    epoch=data_manager.latest_epoch,
                    num_iter=num_iter,
                )
            else:
                ekf.no_measurement()
                print("No measurements made in measurement step")
        else:
            ekf.no_measurement()

        error.append(ekf.r_m - next_state[0:3])
        vel_error.append(ekf.v_m - next_state[3:6])
        ua_error.append(ekf.ua / ua_scale)
        cov_trace.append(np.trace(ekf.P_m))
        gyro_bias_error.append(ekf.w_b / gyro_bias_scale - imu_gyro_bias)
        actual_bias.append(imu_gyro_bias)
        drag_estimate.append(ekf.drag_est)

        sigma_high.append(
            np.array(
                [
                    3 * np.sqrt(ekf.P_m[0, 0]),
                    3 * np.sqrt(ekf.P_m[1, 1]),
                    3 * np.sqrt(ekf.P_m[2, 2]),
                ]
            )
        )
        sigma_low.append(
            np.array(
                [
                    -3 * np.sqrt(ekf.P_m[0, 0]),
                    -3 * np.sqrt(ekf.P_m[1, 1]),
                    -3 * np.sqrt(ekf.P_m[2, 2]),
                ]
            )
        )

    if isinstance(landmark_bearing_sensor, SimulatedMLLandmarkBearingSensor):
        # save measurements to pickle file
        with open(f"od-simulation-data-{time()}.pkl", "wb") as file:
            pickle.dump(data_manager, file)

    # Print final covariance matrix
    print(ekf.P_m)

    plt.plot(error)
    plt.plot(sigma_high, "r--")
    plt.plot(sigma_low, "r--")
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
    plt.title("EKF Unmodelled Acceleration")

    plt.figure()

    plt.plot(gyro_bias_error)
    plt.legend(["x", "y", "z"])
    plt.xlabel("Time step")
    plt.ylabel("Gyro bias error [rad/s]")
    plt.title("EKF Gyro Bias Error")

    plt.figure()

    plt.plot(cov_trace)
    plt.xlabel("Time step")
    plt.ylabel("Covariance trace")
    plt.title("EKF Covariance Trace")

    plt.figure()

    plt.plot(actual_bias)
    plt.legend(["x", "y", "z"])
    plt.xlabel("Time step")
    plt.ylabel("Actual gyro bias [rad/s]")
    plt.title("Actual Gyro Bias")

    plt.figure()
    plt.plot(drag_estimate)
    plt.xlabel("Time step")
    plt.ylabel("Drag estimate")
    plt.title("EKF Drag Estimate")

    plt.show()


if __name__ == "__main__":
    # Run state propagation for the satellite based on ICs
    load_brahe_data_files_if_needed()
    run_simulation()

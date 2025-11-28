"""
Testing the EKF class.
"""

# pylint: disable=import-error
import os
import pickle
from time import time

import brahe
import numpy as np
import quaternion
import scipy as sp
from brahe.epoch import Epoch
from dynamics.ekf_dynamics import EKFDynamics
from dynamics.orbital_dynamics import Dynamics
from sensors.camera_model import CameraModelManager
from sensors.imu import IMU

# from utils.brahe_utils import load_brahe_data_files, load_brahe_data_files_if_needed
from utils.config_utils import load_config
from utils.math_utils import skew
from utils.orbit_utils import get_sso_orbit_state, is_over_daytime

from orbit_determination.ekf import EKF
from orbit_determination.landmark_bearing_sensors import (
    GroundTruthLandmarkBearingSensor,
    SimulatedMLLandmarkBearingSensor,
)
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements, duplicate-code
# mypy: disable-error-code = attr-defined


def run_simulation(trial: int) -> None:
    """
    Run the simulation.

    :return: None
    """

    config = load_config()
    # Set the world update rate and mission duration to a rate that is workable for testing
    config["solver"]["world_update_rate"] = 2  # Hz
    config["mission"]["duration"] = 5 * 90 * 60  # s

    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))  # number of time steps in the simulation

    landmark_bearing_sensor = GroundTruthLandmarkBearingSensor()
    camera_model_manager = CameraModelManager()
    data_manager = ODSimulationDataManager(starting_epoch, dt)

    initial_state = get_sso_orbit_state(starting_epoch, 0, -73, 510e3, northwards=True)
    initial_state = initial_state / 1e3  # Convert from m to km and m/s to km/s
    # Set the initial rotation matrix to identity
    init_rot = np.eye(3)
    # Fix a constant rotation velocity for the test.
    rot = np.array([0, 0, np.pi / 18])

    data_manager.push_next_state(initial_state, init_rot, rot)

    # Set the number of update iterations for the IEKF
    num_iter = 5

    # Set up scaling parameters
    position_scale = 1e-3  # position [km]
    velocity_scale = 1e2  # velocity [km/s]
    ua_scale = 1e8  # unmodelled acceleration [km/s^2]
    drag_scalar_scale = 1  # drag scalar
    axisangle_scale = 1e1  # axis-angle scaling for quaternion
    gyro_bias_scale = 1e2  # gyro bias scaling

    # Define initial uncertainty
    init_pos_std = 100  # km
    init_vel_std = 1e-2  # km/s
    init_ua_std = 5e-8  # km/s^2
    init_drag_std = 3  # drag scalar
    init_quat_std = 1e-3  # axis-angle error [rad]
    init_gyro_bias_std = 1e-2  # 5e-5  # gyro bias error [rad/s]

    # Prep Q matrix for the EKF.
    Q = np.zeros((16, 16))
    # Unmodelled acceleration has larger uncertainty
    Q[6:9, 6:9] = np.eye(3) * 1e-26  #  * 1e-30
    # drag uncertainty is larger
    Q[9, 9] = 1e-4
    # # Bias uncertainty also larger
    Q[13:16, 13:16] = np.eye(3) * 1e-11

    P = np.diag(
        [(init_pos_std) ** 2] * 3  # r
        + [(init_vel_std) ** 2] * 3  # v
        + [(init_ua_std) ** 2] * 3  # ua
        + [(init_drag_std) ** 2]  # drag
        + [(init_quat_std) ** 2] * 3  # quaternion
        + [(init_gyro_bias_std) ** 2] * 3  # gyro bias
    )

    # Set up dynamics instance for ground truth and EKF
    ground_truth_dynamics = Dynamics(
        config=config,
        use_drag=True,
        use_j2=True,
        use_j34=False,
        use_sun_grav=True,
        use_moon_grav=True,
    )
    variable_scaling = np.array(
        [position_scale] * 3  # r
        + [velocity_scale] * 3  # v
        + [ua_scale] * 3  # ua
        + [drag_scalar_scale]  # drag
        + [axisangle_scale] * 3  # quaternion
        + [gyro_bias_scale] * 3  # gyro bias
    )
    ekf_dynamics = EKFDynamics(
        config=config,
        use_drag=False,
        use_j2=True,
        use_j34=False,
        use_sun_grav=False,
        use_moon_grav=False,
        use_unmodelled_a=True,
        use_drag_scalar=True,
    )

    # Initialize IMU and EKF
    imu = IMU.get_default_imu(dt)
    # gyro_bias = imu.get_bias()[0] + np.random.normal(0, init_gyro_bias_std, 3)

    # Apply error to init_rot and ensure orthonormality
    noisy_rot = sp.linalg.expm(skew(np.random.normal(0, init_quat_std, 3)))

    # Assert orthonormality
    assert np.allclose(noisy_rot @ noisy_rot.T, np.eye(3), atol=1e-3) and np.isclose(
        np.linalg.det(noisy_rot), 1
    ), "Rotation matrix is not a proper rotation matrix"

    ekf = EKF(
        # error ranges are in km and km/s
        r=initial_state[0:3] + np.random.normal(0, init_pos_std, 3),
        v=initial_state[3:6] + np.random.normal(0, init_vel_std, 3),
        ua=np.zeros(3),  # np.random.normal(0, init_ua_std, 3),
        q=quaternion.as_float_array(quaternion.from_rotation_matrix(noisy_rot)),
        P=P,
        Q=Q,
        dt=dt,
        config=config,
        ekf_dynamics=ekf_dynamics,
        w_b=np.zeros(3),  # gyro_bias,
        state_scaling=variable_scaling,
    )

    # Store data for plotting
    ekf_state = []
    ekf_state_std = []
    true_state = []
    cov_trace = []
    cov_cond_num = []

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
            x=x[0:6],
            dt=dt,
            epoch=data_manager.latest_epoch,  # pylint: disable=E1136  # pylint/issues/9590
        )
        next_quat = quaternion.from_rotation_matrix(q) * quaternion.from_rotation_vector(w * dt)

        ekf.predict(u=gyro_meas, epoch=data_manager.latest_epoch)

        data_manager.push_next_state(next_state[0:6], quaternion.as_rotation_matrix(next_quat), w)

        if t % 120 == 0 and is_over_daytime(
            data_manager.latest_epoch, data_manager.latest_state[:3] * 1e3
        ):
            for camera_name in CameraModelManager.CAMERA_NAMES:
                data_manager.take_measurement(
                    landmark_bearing_sensor, camera_model_manager[camera_name]
                )
            print(f"Total measurements so far: {data_manager.measurement_count}")
            print(f"Completion: {100 * t / N:.2f}%")
            print(f"State position: {next_state[0:3]}")
            print(
                np.diag(1.0 / ekf.state_scaling[0:3])
                @ ekf.P_m[0:3, 0:3]
                @ np.diag(1.0 / ekf.state_scaling[0:3])
            )

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

        # Check if ekf.P_m is symmetric and positive semidefinite (sdp)
        if not np.allclose(ekf.P_m, ekf.P_m.T, atol=1e-8):
            print(f"Warning: ekf.P_m is not symmetric at step {t}")
            ekf.P_m = (ekf.P_m + ekf.P_m.T) / 2  # temp hack: Make it symmetric
        eigvals = np.linalg.eigvalsh(ekf.P_m)
        if np.any(eigvals < 0):
            print(
                f"Warning: ekf.P_m is not positive semidefinite at step {t}, min eigenvalue: {eigvals.min()}"
            )
        # Check condition number of ekf.P_m
        cond_number = np.linalg.cond(ekf.P_m)
        if cond_number > 1e14:
            print(f"Warning: ekf.P_m condition number is very high at step {t}: {cond_number:.2e}")
        # R_filter = transform_eci_to_lvlh(ekf.r_m, ekf.v_m)
        # filter_position = R_filter @ ekf.r_m
        # gt_position_lvlh = R_filter @ next_state[0:3]

        # pos_error_lvlh.append(filter_position - gt_position_lvlh)

        ekf_x = np.concatenate(
            [
                ekf.r_p,
                ekf.v_p,
                ekf.ua,
                ekf.drag_est,
                ekf.q_p,
                gyro_meas - ekf.w_b,
                ekf.w_b,
            ]
        )
        true_x = np.concatenate(
            [
                next_state[0:6],
                ekf.ekf_dynamics.true_unmodelled_acceleration(
                    x=next_state, epoch=data_manager.latest_epoch
                ),
                ekf.ekf_dynamics.true_drag_constant(
                    x=next_state[0:6], epoch=data_manager.latest_epoch
                ),  # pylint: disable=E1136  # pylint/issues/9590
                next_quat.components,
                w,
                imu_gyro_bias,
            ]
        )

        ekf_state.append(ekf_x)
        ekf_state_std.append((np.diag(ekf.P_m) ** 0.5) / ekf.state_scaling)
        true_state.append(true_x)
        cov_trace.append(np.trace(ekf.P_m))
        cov_cond_num.append(cond_number)
        # [TODO:] store measurement data for plotting

    if isinstance(landmark_bearing_sensor, SimulatedMLLandmarkBearingSensor):
        # save measurements to pickle file
        with open(f"od-simulation-data-{time()}.pkl", "wb") as file:
            pickle.dump(data_manager, file)

    # Print final covariance matrix
    print(np.diag(1.0 / ekf.state_scaling) @ ekf.P_m @ np.diag(1.0 / ekf.state_scaling))

    # Create three subplots for x,y,z position error
    ekf_state_np = np.array(ekf_state)
    ekf_state_std_np = np.array(ekf_state_std)
    true_state_np = np.array(true_state)
    cov_trace = np.array(cov_trace)
    cov_cond_num = np.array(cov_cond_num)

    index_trial = trial + 0
    dir_name = f"results/ekf_realdrag/trial_{index_trial}"
    os.makedirs(dir_name, exist_ok=True)
    np.save(os.path.join(dir_name, "ekf_state.npy"), ekf_state_np)
    np.save(os.path.join(dir_name, "ekf_state_std.npy"), ekf_state_std_np)
    np.save(os.path.join(dir_name, "true_state.npy"), true_state_np)
    np.save(os.path.join(dir_name, "cov_trace.npy"), cov_trace)
    np.save(os.path.join(dir_name, "cov_cond_num.npy"), cov_cond_num)


if __name__ == "__main__":
    # Run state propagation for the satellite based on ICs
    # load_brahe_data_files_if_needed()
    #### load_brahe_data_files()
    TRIALS = 1
    # np.random.seed(69420)
    for i in range(TRIALS):
        run_simulation(i)

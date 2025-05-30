"""
Run the EKF using CLI.

This script requires the following arguments:
    --name: Name of the experiment
    --meas_rate: Rate at which measurements are taken

The script expects to find the following contents in the output directory:
- /output_dir
    - /experiment_name
        - ground_truth.npz
            - trajectory
            - attitude
            - daytime

The script will generate (append in the args.json case) the following contents in the output directory:
- /output_dir
    - /experiment_name
        - args.json
        - /ekf_data
            - pos_state
            - vel_state
            - ua_state
            - pos_cov_trace
            - gyro_bias_state
            - actual_bias
            - drag_scalar_state

"""

import argparse
import json
import os
import pickle
import subprocess
from time import time

import brahe
import numpy as np
import quaternion
# from brahe import Epoch

from dynamics.ekf_dynamics import EKFDynamics
from orbit_determination.ekf import EKF
from orbit_determination.landmark_bearing_sensors import (
    GroundTruthLandmarkBearingSensor,
    SimulatedMLLandmarkBearingSensor,
    SimulatedMLStoredLandmarkBearingSensor,
)
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from sensors.camera_model import CameraModelManager
from sensors.imu import IMU
from utils.brahe_utils import load_brahe_data_files_if_needed
from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.orbit_utils import is_over_daytime

# pylint: disable=too-many-locals


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run the EKF simulation.")
    parser.add_argument(
        "--name",
        type=str,
        default="test",
        help="Name of the experiment",
    )

    return parser.parse_args()


def run_simulation(args) -> None:
    """
    Run the simulation.

    :param args: The command line arguments.

    :return: None
    """

    user_config = load_config(USER_CONFIG_PATH)
    output_basedir = os.path.join(user_config["output_dir"], args.name)
    # Load json
    try:
        with open(os.path.join(output_basedir,"args.json"),"r") as jsonfile:
            arg_data = json.load(jsonfile)

    except Exception as e:
        raise ValueError(f"Error in args.json in {args.name}: {e}")

    # Check that if a name was provided it matches the one in the json file
    assert (
        arg_data["name"] == args.name
    ), f"Name in args.json does not match the provided name: {arg_data['name']} != {args.name}"

    f = arg_data["frequency"]
    mission_duration = arg_data["duration"]
    angular_velocity = arg_data["angular_velocity"]
    meas_rate = arg_data["meas_rate"]

    # No new args to be stored.

    # Set the world update rate and mission duration
    # Technically we don't need to set these because we get them from the json file
    # but we need the config for the EKF so we might as well set it here
    config = load_config()
    config["solver"]["world_update_rate"] = f  # Hz
    config["mission"]["duration"] = mission_duration  # s

    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = brahe.Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))  # number of time steps in the simulation

    landmark_bearing_sensor = SimulatedMLStoredLandmarkBearingSensor(output_basedir)
    camera_model_manager = CameraModelManager()
    data_manager = ODSimulationDataManager(starting_epoch, dt)

    data_dir = os.path.join(output_basedir, "ground_truth.npz")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Ground truth data file {data_dir} does not exist."
        )
    data = np.load(data_dir)
    trajectory_gt = data["trajectory"]
    attitude_gt = data["attitude"]
    daytime_gt = data["daytime"]

    # Set the initial rotation matrix to identity

    data_manager.push_next_state(trajectory_gt[0], attitude_gt[0])

    # Apply error to init_rot and ensure orthonormality
    noisy_rot = attitude_gt[0] + np.random.normal(0, 1e-2, (3, 3))
    noisy_rot = noisy_rot @ np.linalg.inv(np.linalg.cholesky(noisy_rot.T @ noisy_rot))

    # Assert orthonormality
    assert np.allclose(noisy_rot @ noisy_rot.T, np.eye(3), atol=1e-2) and np.isclose(
        np.linalg.det(noisy_rot), 1
    ), "Rotation matrix is not a proper rotation matrix"

    # Set the number of update iterations for the IEKF
    num_iter = 5

    # Set up scaling parameter for the unmodelled acceleration
    ua_scale = 10

    # Set up scaling parameter for gyro bias
    gyro_bias_scale = 2

    # Fix a constant rotation velocity for the test.
    rot = np.array(angular_velocity)

    # Prep Q matrix for the EKF.
    Q = np.eye(16) * 1e-12
    # Unmodelled acceleration has larger uncertainty
    Q[6:9, 6:9] = np.eye(3) * 1e-9
    # Bias uncertainty also larger
    Q[13:16, 13:16] = np.eye(3) * 1e-9

    P = np.eye(16)
    P[0:3, 0:3] *= 5
    P[3:6, 3:6] *= 5
    P[6:9, 6:9] *= 1e-4
    P[9:10, 9:10] *= 1e-4
    P[10:13, 10:13] *= 1e-4
    P[13:16, 13:16] *= 1e-4

    ekf_dynamics = EKFDynamics(
        config=config,
        use_drag=False,
        use_j2=False,
        use_j34=False,
        use_unmodelled_a=True,
        use_drag_scalar=True,
        use_moon_grav=False,
        use_sun_grav=False,
        ua_scale=ua_scale,
    )

    # Initialize IMU and EKF
    imu = IMU.get_default_imu(dt)
    gyro_bias = (imu.get_bias()[0] + np.random.normal(0, 5e-5, 3)) * gyro_bias_scale
    ekf = EKF(
        # error ranges are in meters and m/s
        r=trajectory_gt[0][0:3] + np.random.normal(0, 5000, 3),
        v=trajectory_gt[0][3:6] + np.random.normal(0, 10, 3),
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

    pos_state = []
    vel_state = []
    ua_state = []
    pos_cov_trace = []
    gyro_bias_state = []
    actual_bias = []
    drag_scalar_state = []


    for t in range(0, N - 1):
        # Apply noise to x, y to generate angular wobble around the primary rotation axis z
        # One rotation every 10 seconds to model a relatively slow wobble
        w = rot + 0.05 * np.array(
            [np.cos(2 * np.pi * t / (10 / dt)), np.sin(2 * np.pi * t / (10 / dt)), 0]
        )

        # Get a gyro measurement to use in the EKF and the current gyro bias for the ground truth
        gyro_meas, _ = imu.update(w, np.zeros((3)))
        imu_gyro_bias = imu.get_bias()[0]

        ekf.predict(u=gyro_meas, epoch=data_manager.latest_epoch)
        data_manager.push_next_state(trajectory_gt[t], attitude_gt[t])

        # if t % meas_rate == 0 and is_over_daytime(
        #     data_manager.latest_epoch, data_manager.latest_state[:3]
        # ):
        if t % meas_rate == 0 and daytime_gt[t]:
            # Generate vision inference measurements
            subprocess.run(
                [
                    "python3",
                    "scripts/run_vision.py",
                    "--name",
                    args.name,
                    "--timestep",
                    str(t),
                ]
            )
            
            # Take measurements with the landmark bearing sensor
            for camera_name in CameraModelManager.CAMERA_NAMES:
                data_manager.take_measurement(
                    landmark_bearing_sensor, camera_model_manager[camera_name]
                )
            print(f"Total measurements so far: {data_manager.measurement_count}")
            print(f"Completion: {100 * t / N:.2f}%")

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

        pos_state.append(ekf.r_m)
        vel_state.append(ekf.v_m)
        ua_state.append(ekf.ua/ua_scale)
        pos_cov_trace.append(np.trace(ekf.P_m[0:3,0:3]))
        gyro_bias_state.append(ekf.w_b / gyro_bias_scale)
        actual_bias.append(imu_gyro_bias)
        drag_scalar_state.append(ekf.drag_est)

    pos_state = np.array(pos_state)
    vel_state = np.array(vel_state)
    ua_state = np.array(ua_state)
    pos_cov_trace = np.array(pos_cov_trace)
    gyro_bias_state = np.array(gyro_bias_state)
    actual_bias = np.array(actual_bias)
    drag_scalar_state = np.array(drag_scalar_state)

    ekf_dir = os.path.join(output_basedir,"ekf_data")
    os.makedirs(ekf_dir, exist_ok=True)
    np.save(os.path.join(ekf_dir,"pos_state.npy"),pos_state)
    np.save(os.path.join(ekf_dir,"vel_state.npy"),vel_state)
    np.save(os.path.join(ekf_dir,"ua_state.npy"),ua_state)
    np.save(os.path.join(ekf_dir,"pos_cov_trace.npy"),pos_cov_trace)
    np.save(os.path.join(ekf_dir,"gyro_bias_state.npy"),gyro_bias_state)
    np.save(os.path.join(ekf_dir,"actual_bias.npy"),actual_bias)
    np.save(os.path.join(ekf_dir,"drag_scalar_state.npy"),drag_scalar_state)

    if isinstance(landmark_bearing_sensor, SimulatedMLLandmarkBearingSensor):
        # save measurements to pickle file
        with open(f"od-simulation-data-{time()}.pkl", "wb") as file:
            pickle.dump(data_manager, file)


if __name__ == "__main__":
    args = parse_args()
    load_brahe_data_files_if_needed()
    run_simulation(args)

"""
Run the EKF using CLI.

This script requires the following arguments:
    -f: Frequency of the spacecraft trajectory
    --mission_duration: Duration of the spacecraft mission for which we are generating the trajectory [s]
    --name: Name of the experiment
    --angular_velocity: Angular velocity of the spacecraft [rad/s]
    --meas_rate: Rate at which measurements are taken

The script expects to find the following contents in the output directory:
- /output_dir
    - /experiment_name
        - trajectory_gt.npy
        - attitude_gt.npy

The script will generate the following contents in the output directory:
- /output_dir
    - /experiment_name
        - ekf_state_data_.pkl

The state data contains the following fields:
    - timestep: The time step
    - prior_position: The prior position estimate
    - prior_velocity: The prior velocity estimate
    - prior_attitude: The prior attitude estimate
    - prior_covariance: The prior covariance estimate
    - posterior_position: The posterior position estimate
    - posterior_velocity: The posterior velocity estimate
    - posterior_attitude: The posterior attitude estimate
    - posterior_covariance: The posterior covariance estimate
    - gyro_bias_estimate: The estimated gyro bias
    - gyro_bias: The actual gyro bias
    - unmodelled_acceleration: The unmodelled acceleration estimate

"""

import os
import pickle
from argparse import ArgumentParser
from time import time

import brahe
import numpy as np
import quaternion
from brahe.epoch import Epoch

from dynamics.ekf_dynamics import EKFDynamics
from orbit_determination.ekf import EKF
from orbit_determination.landmark_bearing_sensors import (
    GroundTruthLandmarkBearingSensor,
    SimulatedMLLandmarkBearingSensor,
)
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from sensors.camera_model import CameraModelManager
from utils.brahe_utils import load_brahe_data_files_if_needed
from utils.config_utils import load_config
from utils.imu_utils import imu_init
from utils.orbit_utils import is_over_daytime

# pylint: disable=too-many-locals


def run_simulation(args) -> None:
    """
    Run the simulation.

    :param args: The command line arguments.

    :return: None
    """

    config = load_config()
    # Set the world update rate and mission duration to a rate that is workable for testing
    config["solver"]["world_update_rate"] = args.f  # Hz
    config["mission"]["duration"] = args.mission_duration  # s

    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))  # number of time steps in the simulation

    landmark_bearing_sensor = GroundTruthLandmarkBearingSensor()
    camera_model_manager = CameraModelManager()
    data_manager = ODSimulationDataManager(starting_epoch, dt)

    if not os.path.exists(f"output_dir/{args.name}/trajectory_gt.npy") or not os.path.exists(
        f"output_dir/{args.name}/attitude_gt.npy"
    ):
        raise FileNotFoundError(
            f"One of the required files in {args.name} does not exist. Please run the trajectory generation script first."
        )

    trajectory_gt = np.load(f"output_dir/{args.name}/trajectory_gt.npy")
    attitude_gt = np.load(f"output_dir/{args.name}/attitude_gt.npy")
    # Set the initial rotation matrix to identity

    data_manager.push_next_state(trajectory_gt[0], attitude_gt[0])

    # Apply error to init_rot and ensure orthonormality
    noisy_rot = attitude_gt[0] + np.random.normal(0, 1e-2, (3, 3))
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
    rot = np.array(args.angular_velocity)

    # Prep Q matrix for the EKF.
    Q = np.eye(15) * 1e-12
    # Unmodelled acceleration has larger uncertainty
    Q[6:9, 6:9] = np.eye(3) * 1e-9
    # Bias uncertainty also larger
    Q[12:15, 12:15] = np.eye(3) * 1e-9

    P = np.eye(15)
    P[0:3, 0:3] *= 5
    P[3:6, 3:6] *= 5
    P[6:9, 6:9] *= 1e-4
    P[9:12, 9:12] *= 1e-4
    P[12:15, 12:15] *= 1e-4

    ekf_dynamics = EKFDynamics(
        config=config,
        use_drag=False,
        use_j2=False,
        use_unmodelled_a=True,
        ua_scale=ua_scale,
    )

    # Initialize IMU and EKF
    imu = imu_init(dt)
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

        if t % args.meas_rate == 0 and is_over_daytime(
            data_manager.latest_epoch, data_manager.latest_state[:3]
        ):
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

        state_data = {
            "timestep": t,
            "prior_position": ekf.r_p,
            "prior_velocity": ekf.v_p,
            "prior_attitude": ekf.q_p,
            "prior_covariance": ekf.P_p,
            "posterior_position": ekf.r_m,
            "posterior_velocity": ekf.v_m,
            "posterior_attitude": ekf.q_m,
            "posterior_covariance": ekf.P_m,
            "gyro_bias_estimate": ekf.w_b,
            "gyro_bias": imu_gyro_bias,
            "unmodelled_acceleration": ekf.ua,
        }
        # Save the state data to a file
        with open(f"output_dir/{args.name}/ekf_state_data_.pkl", "ab") as file:
            pickle.dump(state_data, file)

    if isinstance(landmark_bearing_sensor, SimulatedMLLandmarkBearingSensor):
        # save measurements to pickle file
        with open(f"od-simulation-data-{time()}.pkl", "wb") as file:
            pickle.dump(data_manager, file)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run the EKF simulation.")

    parser.add_argument(
        "--f",
        type=float,
        default="1",
        help="Frequency of the spacecraft trajectory",
    )
    parser.add_argument(
        "--mission_duration",
        type=float,
        default="2700",
        help="Duration of the spacecraft mission for which we are generating the trajectory [s]",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="test",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--angular_velocity",
        type=list,
        default=[0, 0, np.pi / 18],
        help="Angular velocity of the spacecraft [rad/s]",
    )
    parser.add_argument(
        "--meas_rate",
        type=int,
        default=120,
        help="The rate at which measurements are supposed to be taken. 120 means that a measurement"
        "is taken every 120 timesteps",
    )

    args = parser.parse_args()

    load_brahe_data_files_if_needed()
    run_simulation(args)

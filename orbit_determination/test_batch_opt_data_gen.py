"""
Testing the batch opt data generation via the EKF class.
We need 
"""

import os
import pickle
from time import time

import brahe
import h5py
import numpy as np
import quaternion
from brahe.epoch import Epoch

# from dynamics.orbital_dynamics import Dynamics
from dynamics.orbital_att_dynamics import Dynamics
from dynamics.orbital_att_dynamics import DynamicsIDX as dynidx
from sensors.camera_model import CameraModelManager
from sensors.imu import IMU
from utils.brahe_utils import load_brahe_data_files_if_needed
from utils.config_utils import load_config
from utils.orbit_utils import get_sso_orbit_state, is_over_daytime

from orbit_determination.landmark_bearing_sensors import (
    GroundTruthLandmarkBearingSensor,
    SimulatedMLLandmarkBearingSensor,
)
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from orbit_determination.testing.plot_batch_opt_test_data import plot_syn_data

# pylint: disable=too-many-locals


def run_simulation(trial) -> None:
    """
    Run the simulation.

    :return: None
    """
    idx = dynidx(has_gyro_bias=True)
    config = load_config()
    # Set the world update rate and mission duration to a rate that is workable for testing
    config["solver"]["world_update_rate"] = 2  # Hz
    config["mission"]["duration"] = 100  # s

    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))  # number of time steps in the simulation

    landmark_bearing_sensor = GroundTruthLandmarkBearingSensor()
    camera_model_manager = CameraModelManager()
    data_manager = ODSimulationDataManager(starting_epoch, dt, idx)

    initial_state = np.zeros((idx.NX,))
    initial_state[idx.ORB] = get_sso_orbit_state(starting_epoch, 0, -73, 580e3, northwards=True)
    initial_state[idx.ORB] = initial_state[idx.ORB] / 1e3  # Convert from m to km and m/s to km/s

    # Set the initial rotation matrix to identity
    init_rot = np.eye(3)

    # Apply error to init_rot and ensure orthonormality
    noisy_rot = init_rot + np.random.normal(0, 1e-3, (3, 3))
    noisy_rot = noisy_rot @ np.linalg.inv(np.linalg.cholesky(noisy_rot.T @ noisy_rot))

    # Assert orthonormality
    assert np.allclose(noisy_rot @ noisy_rot.T, np.eye(3), atol=1e-3) and np.isclose(
        np.linalg.det(noisy_rot), 1
    ), "Rotation matrix is not a proper rotation matrix"

    # Fix a constant rotation velocity for the test.
    rot = np.array([0, 0, np.pi / 18])

    initial_att = quaternion.from_rotation_matrix(noisy_rot)
    initial_state[idx.QUAT] = quaternion.as_float_array(initial_att)
    initial_state[idx.OMEGA] = rot

    initial_state[idx.GYR_BIAS] = np.random.normal(0, config["satellite"]["gyro"]["bias_std"], (3,))

    data_manager.push_next_state(initial_state)

    # Set up dynamics instance for ground truth and EKF
    ground_truth_dynamics = Dynamics(
        config=config,
        use_drag=True,
        use_j2=True,
        use_j34=False,
        use_sun_grav=True,
        use_moon_grav=True,
        include_gyro_bias=True,
        gyro_bias_tau=config["satellite"]["gyro"]["bias_tau"],
        gyro_bias_std=config["satellite"]["gyro"]["bias_std"],
    )

    # Initialize IMU and EKF
    imu_dt = 1.0 / config["satellite"]["gyro"]["sampling_rate"]
    last_imu = -np.inf
    imu = IMU.get_default_imu(imu_dt)

    all_landmark_measurements = np.zeros((0, 7))
    all_gyro_measurements = np.zeros((0, 4))
    all_landmark_group_start = np.array([])
    last_vis = -np.inf
    vis_dt = 10.0

    # init_gyro_meas, _ = imu.update(initial_state[dynidx.OMEGA], np.zeros((3)))

    init_gyro_meas = initial_state[idx.OMEGA] + initial_state[idx.GYR_BIAS]
    init_gyro_meas += np.random.normal(0, config["satellite"]["gyro"]["noise_density"], (3,))

    gyro_measurement = np.concatenate(
        [np.array([starting_epoch.to_datetime().timestamp()]), init_gyro_meas]
    )
    all_gyro_measurements = np.vstack([all_gyro_measurements, gyro_measurement])

    # measurements need to be properly timestamped
    epochs_list = starting_epoch.to_datetime().timestamp() + np.arange(N) * data_manager.dt

    for i in range(0, N - 1):
        t = epochs_list[i]
        # take a set of measurements every minute
        x = data_manager.latest_state

        next_state = ground_truth_dynamics.perturbed_f(x=x, dt=dt, epoch=data_manager.latest_epoch)

        data_manager.push_next_state(next_state)

        # Get a gyro measurement to use in the EKF and the current gyro bias for the ground truth
        if last_imu + imu_dt <= t:
            # gyro_meas, _ = imu.update(x[dynidx.OMEGA], np.zeros((3)))
            gyro_meas = x[idx.OMEGA] + x[idx.GYR_BIAS]
            gyro_meas += np.random.normal(0, config["satellite"]["gyro"]["noise_density"], (3,))
            gyro_measurement = np.concatenate([np.array([t]), gyro_meas])
            all_gyro_measurements = np.vstack([all_gyro_measurements, gyro_measurement])
            last_imu = t

        if last_vis + vis_dt <= t:
            # and is_over_daytime(
            #     data_manager.latest_epoch, data_manager.latest_state[:3] * 1e3
            # ):
            for camera_name in CameraModelManager.CAMERA_NAMES:
                data_manager.take_measurement(
                    landmark_bearing_sensor, camera_model_manager[camera_name]
                )
            print(f"Completion: {100 * i / N:.2f}%")

            _, *z = data_manager.latest_measurements

            if z[0].shape[0] > 0:
                z1 = np.concatenate(
                    [z[0], z[1]], axis=1
                )  # Concatenate bearing unit vectors and landmarks
                tmp = np.expand_dims(np.array([t] * z[0].shape[0]), axis=1)
                landmark_measurement = np.concatenate([tmp, z1], axis=1)

                landmark_group_start = np.array([0] * z[0].shape[0])
                landmark_group_start[0] = 1  # Mark the first measurement in the group

                all_landmark_measurements = np.vstack(
                    [all_landmark_measurements, landmark_measurement]
                )
                all_landmark_group_start = np.concatenate(
                    [all_landmark_group_start, landmark_group_start]
                )
                print(
                    f"Measurement at epoch {data_manager.latest_epoch} with {z[0].shape[0]} landmarks"
                )
            last_vis = t

    if isinstance(landmark_bearing_sensor, SimulatedMLLandmarkBearingSensor):
        # save measurements to pickle file
        with open(f"od-simulation-data-{time()}.pkl", "wb") as file:
            pickle.dump(data_manager, file)

    dir_name = f"batch_opt_gen"
    os.makedirs(dir_name, exist_ok=True)

    # Ensure it's a 2D array for saving
    all_landmark_group_start = np.expand_dims(all_landmark_group_start, axis=1)

    with h5py.File(f"{dir_name}/orbit_measurements.h5", "w") as f:
        # top-level datasets
        f.create_dataset("landmark_measurements", data=all_landmark_measurements)
        f.create_dataset("gyro_measurements", data=all_gyro_measurements)
        # TODO: store measurement covariances

        f.create_dataset("group_starts", data=all_landmark_group_start)

    # Save ground truth states for reference
    with h5py.File(f"{dir_name}/ground_truth_states.h5", "w") as f:
        f.create_dataset("states", data=data_manager.states)
        f.create_dataset("unixtime", data=epochs_list)

    plot_syn_data(
        epochs_list, data_manager.states, all_landmark_measurements, all_gyro_measurements, dir_name
    )


if __name__ == "__main__":
    # Run state propagation for the satellite based on ICs
    load_brahe_data_files_if_needed()
    trials = 1
    # np.random.seed(69420)
    for i in range(trials):
        run_simulation(i)

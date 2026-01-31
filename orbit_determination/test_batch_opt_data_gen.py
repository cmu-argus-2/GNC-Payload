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


from dynamics.orbital_dynamics import Dynamics
from orbit_determination.landmark_bearing_sensors import (
    GroundTruthLandmarkBearingSensor,
    SimulatedMLLandmarkBearingSensor,
)
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from sensors.camera_model import CameraModelManager
from sensors.imu import IMU
from utils.brahe_utils import load_brahe_data_files_if_needed
from utils.config_utils import load_config
from utils.orbit_utils import get_sso_orbit_state, is_over_daytime

# pylint: disable=too-many-locals

def enforce_quat_continuity(next_quat, prev_quat, i):
    # components is [w, x, y, z]
    a = prev_quat.components
    b = next_quat.components
    if np.dot(a, b) < 0.0:
        next_quat = -next_quat
        print(f"Sign flipped at timestep {i}!")
    return next_quat

def run_simulation(trial) -> None:
    """
    Run the simulation.

    :return: None
    """

    config = load_config()
    # Set the world update rate and mission duration to a rate that is workable for testing
    config["solver"]["world_update_rate"] = 2  # Hz
    config["mission"]["duration"] = 100  # s

    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))  # number of time steps in the simulation

    landmark_bearing_sensor = GroundTruthLandmarkBearingSensor()
    camera_model_manager = CameraModelManager()
    data_manager = ODSimulationDataManager(starting_epoch, dt)

    initial_state = get_sso_orbit_state(starting_epoch, 0, -73, 600e3, northwards=True)
    initial_state = initial_state / 1e3  # Convert from m to km and m/s to km/s
    # Set the initial rotation matrix to identity
    init_rot = np.eye(3)

    data_manager.push_next_state(initial_state, init_rot)

    # Apply error to init_rot and ensure orthonormality
    noisy_rot = init_rot + np.random.normal(0, 1e-3, (3, 3))
    noisy_rot = noisy_rot @ np.linalg.inv(np.linalg.cholesky(noisy_rot.T @ noisy_rot))

    # Assert orthonormality
    assert np.allclose(noisy_rot @ noisy_rot.T, np.eye(3), atol=1e-3) and np.isclose(
        np.linalg.det(noisy_rot), 1
    ), "Rotation matrix is not a proper rotation matrix"

    # Fix a constant rotation velocity for the test.
    rot = np.array([np.pi / 12, np.pi / 6, np.pi / 18])

    # Prep Q matrix for the EKF.
    Q = np.eye(16) * 1e-16
    # Unmodelled acceleration has larger uncertainty
    Q[6:9, 6:9] = np.eye(3) * 1e-12
    # # Bias uncertainty also larger
    Q[13:16, 13:16] = np.eye(3) * 1e-12

    P = np.eye(16)
    P[0:3, 0:3] *= 5e-4
    P[3:6, 3:6] *= 5e-4
    P[6:9, 6:9] *= 1e-5
    P[9:10, 9:10] *= 1e-5
    P[10:13, 10:13] *= 1e-5
    P[13:16, 13:16] *= 1e-5

    # Set up dynamics instance for ground truth and EKF
    ground_truth_dynamics = Dynamics(
        config=config,
        use_drag=True,
        use_j2=True,
        use_j34=False,
        use_sun_grav=True,
        use_moon_grav=True,
    )

    # Initialize IMU and EKF
    imu = IMU.get_default_imu(dt)

    all_landmark_measurements = np.zeros((0,7))
    all_gyro_measurements = np.zeros((0,4))
    all_landmark_group_start = np.array([])

    init_gyro_meas, _ = imu.update(rot, np.zeros((3)))
    gyro_measurement = np.concatenate([np.array([starting_epoch.to_datetime().timestamp()]), init_gyro_meas])
    all_gyro_measurements = np.vstack([all_gyro_measurements, gyro_measurement])
    
    # measurements need to be properly timestamped
    epochs_list = starting_epoch.to_datetime().timestamp() + np.arange(N) * data_manager.dt
    quaternions = np.array([[1,0,0,0]])
    for i in range(0, N - 1):
        t = epochs_list[i]
        # take a set of measurements every minute
        x = data_manager.latest_state
        q = data_manager.latest_attitude

        # Apply noise to x, y to generate angular wobble around the primary rotation axis z
        # One rotation every 10 seconds to model a relatively slow wobble
        w = rot #+ 0.05 * np.array(
        #     [np.cos(2 * np.pi * t / (10 / dt)), np.sin(2 * np.pi * t / (10 / dt)), 0]
        # )

        # Get a gyro measurement to use in the EKF and the current gyro bias for the ground truth
        gyro_meas, _ = imu.update(w, np.zeros((3)))

        next_state = ground_truth_dynamics.perturbed_f(
            x=x[0:6], dt=dt, epoch=data_manager.latest_epoch
        )
        q = quaternion.as_quat_array(quaternions[-1,:])
        # next_quat = quaternion.from_rotation_matrix(q) * quaternion.from_rotation_vector(w * dt)
        next_quat = q * quaternion.from_rotation_vector(w * dt)
        next_quat = enforce_quat_continuity(next_quat=next_quat, prev_quat=q,i=i)

        next_quat_store = np.array([quaternion.as_float_array(next_quat)])
        next_quat_store /= np.linalg.norm(next_quat_store)
        quaternions = np.append(quaternions, next_quat_store,axis=0)
        data_manager.push_next_state(next_state[0:6], quaternion.as_rotation_matrix(next_quat))

        if i % 40 == 0:
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

                z1 = np.concatenate([z[0], z[1]], axis=1)  # Concatenate bearing unit vectors and landmarks
                tmp = np.expand_dims(np.array([t] * z[0].shape[0]),axis=1)
                landmark_measurement = np.concatenate([tmp, z1], axis=1)

                landmark_group_start = np.array([0] * z[0].shape[0])
                landmark_group_start[0] = 1  # Mark the first measurement in the group

                all_landmark_measurements = np.vstack([all_landmark_measurements, landmark_measurement])
                all_landmark_group_start = np.concatenate([all_landmark_group_start, landmark_group_start])
                print(f"Measurement at epoch {data_manager.latest_epoch} with {z[0].shape[0]} landmarks")

        # Do the gyro measurement update in any case
        gyro_measurement = np.concatenate([np.array([t+1]), gyro_meas])
        all_gyro_measurements = np.vstack([all_gyro_measurements, gyro_measurement])

    if isinstance(landmark_bearing_sensor, SimulatedMLLandmarkBearingSensor):
        # save measurements to pickle file
        with open(f"od-simulation-data-{time()}.pkl", "wb") as file:
            pickle.dump(data_manager, file)

    dir_name = f"batch_opt_gen"
    os.makedirs(dir_name, exist_ok=True)

    all_landmark_group_start = np.expand_dims(all_landmark_group_start, axis=1)  # Ensure it's a 2D array for saving

    with h5py.File(f"{dir_name}/orbit_measurements.h5", 'w') as f:
        # top-level datasets
        f.create_dataset('landmark_measurements', data=all_landmark_measurements)
        f.create_dataset('gyro_measurements',     data=all_gyro_measurements)
        f.create_dataset('group_starts',          data=all_landmark_group_start)
    
    # Save ground truth states for reference
    states_hist = data_manager.states
    eci_Rs_body_hist = data_manager.eci_Rs_body
    # convert to quaternions for easier storage
    attitudes_hist = quaternions
    omega_hist = np.tile(w, (states_hist.shape[0], 1))  # repeat w for each timestep
    full_state_hist = np.hstack((states_hist, attitudes_hist, omega_hist))
    with h5py.File(f"{dir_name}/ground_truth_states.h5", 'w') as f:
        f.create_dataset('states', data=full_state_hist)
        f.create_dataset('unixtime', data=epochs_list)

    # np.save(os.path.join(dir_name, "pos_error.npy"), error)

if __name__ == "__main__":
    # Run state propagation for the satellite based on ICs
    load_brahe_data_files_if_needed()
    trials = 1
    # np.random.seed(69420)
    for i in range(trials):
        run_simulation(i)

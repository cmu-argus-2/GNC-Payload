from typing import Any
from time import perf_counter
from time import time
import yaml
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

import brahe
from brahe.epoch import Epoch

from dynamics.orbital_dynamics import f
from orbit_determination.landmark_bearing_sensors import (
    RandomLandmarkBearingSensor,
    SimulatedMLLandmarkBearingSensor,
)
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from orbit_determination.nonlinear_least_squares_od import OrbitDetermination

from utils.orbit_utils import get_sso_orbit_state, is_over_daytime


def load_config() -> dict[str, Any]:
    """
    Load the configuration file and modify it for the purposes of this test.

    :return: The modified configuration file as a dictionary.
    """
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # TODO: move this into the config file itself
    # decrease world update rate since we only care about position dynamics
    config["solver"]["world_update_rate"] = 1 / 60  # Hz
    config["mission"]["duration"] = 3 * 90 * 60  # s, roughly 1 orbit

    return config


# TODO: consolidate this with the function in utils/earth_utils.py
def get_nadir_rotation(cubesat_position: np.ndarray) -> np.ndarray:
    """
    Get the rotation matrix from the body frame to the ECI frame for a satellite with an orbital angular momentum in the -y direction.

    :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
    :return: A numpy array of shape (3, 3) containing the rotation matrix from the body frame to the ECI frame.
    """
    y_axis = [0, -1, 0]  # along orbital angular momentum
    z_axis = -cubesat_position / np.linalg.norm(cubesat_position)  # along radial vector
    x_axis = np.cross(y_axis, z_axis)
    R_body_to_eci = np.column_stack([x_axis, y_axis, z_axis])
    return R_body_to_eci


def get_SO3_noise_matrices(N: int, magnitude_std: float) -> np.ndarray:
    """
    Generate a set of matrices representing random rotations in SO(3) with a given standard deviation.

    :param N: The number of noise matrices to generate.
    :param magnitude_std: The standard deviation of the magnitudes of the rotations in radians.
    :return: A numpy array of shape (N, 3, 3) containing the noise rotations.
    """
    magnitudes = np.abs(np.random.normal(scale=magnitude_std, size=N))
    directions = np.random.normal(size=(N, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return Rotation.from_rotvec(magnitudes[:, np.newaxis] * directions).as_matrix()


def test_od():
    # update_brahe_data_files()
    config = load_config()

    # set up simulation parameters
    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))  # number of time steps in the simulation

    # landmark_bearing_sensor = RandomLandmarkBearingSensor(config)
    landmark_bearing_sensor = SimulatedMLLandmarkBearingSensor(config)
    data_manager = ODSimulationDataManager(starting_epoch, dt)
    od = OrbitDetermination(dt)

    # pick a latitude and longitude that results in the satellite passing over the contiguous US in its first few orbits
    initial_state = get_sso_orbit_state(starting_epoch, 0, -73, 600e3, northwards=True)
    data_manager.push_next_state(initial_state, get_nadir_rotation(initial_state[:3]))

    for t in range(0, N - 1):
        # take a set of measurements every 5 minutes
        if t % 5 == 0 and is_over_daytime(data_manager.latest_epoch, data_manager.latest_state[:3]):
            data_manager.take_measurement(landmark_bearing_sensor)
            print(f"Total measurements so far: {data_manager.measurement_count}")
            print(f"Completion: {100 * t / N:.2f}%")

        next_state = f(data_manager.latest_state, dt)
        data_manager.push_next_state(next_state, get_nadir_rotation(next_state[:3]))

    if data_manager.measurement_count == 0:
        raise ValueError("No measurements taken")
    print(f"Total measurements: {data_manager.measurement_count}")

    if type(landmark_bearing_sensor) == SimulatedMLLandmarkBearingSensor:
        # save measurements to pickle file
        with open(f"od-simulation-data-{time()}.pkl", "wb") as file:
            pickle.dump(data_manager, file)

    # for i, attitude_noise in enumerate(attitude_noises):
    #     so3_noise_matrices = get_SO3_noise_matrices(len(times), np.deg2rad(attitude_noise))
    #     bearing_unit_vectors = np.einsum("ijk,ik->ij", so3_noise_matrices, bearing_unit_vectors)
    #
    #     start_time = perf_counter()
    #     estimated_states = od.fit_orbit(times, landmarks, bearing_unit_vectors, Rs_body_to_eci, N)
    #     print(f"Elapsed time: {perf_counter() - start_time:.2f} s")
    #
    #     position_errors = np.linalg.norm(states[:, :3] - estimated_states[:, :3], axis=1)
    #     rms_position_error = np.sqrt(np.mean(position_errors ** 2))
    #     print(f"Attitude SO(3) Noise Variance: {attitude_noise}")
    #     print(f"RMS position error: {rms_position_error}")
    #     print(f"Completion Percentage: {100 * (i + 1) / len(attitude_noises):.2f}%")
    #
    #     rms_position_errors[i] = rms_position_error
    #
    # print(f"{rms_position_errors=}")
    # plt.figure()
    # plt.xlabel("Attitude SO(3) Noise Variance [deg]")
    # plt.ylabel("OD RMS Position Error [km]")
    # plt.title("Effect of Attitude Noise on Orbit Determination")
    # plt.scatter(np.rad2deg(attitude_noises), rms_position_errors / 1e3)
    # plt.plot([0, 10], [50, 50], linestyle="--", color="r")
    # plt.show()

    start_time = perf_counter()
    estimated_states = od.fit_orbit(data_manager)
    print(f"Elapsed time: {perf_counter() - start_time:.2f} s")

    position_errors = np.linalg.norm(data_manager.states[:, :3] - estimated_states[:, :3], axis=1)
    rms_position_error = np.sqrt(np.mean(position_errors**2))
    print(f"RMS position error: {rms_position_error}")

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(-1.5 * R_EARTH, 1.5 * R_EARTH)
    # ax.set_ylim(-1.5 * R_EARTH, 1.5 * R_EARTH)
    # ax.set_zlim(-1.5 * R_EARTH, 1.5 * R_EARTH)
    #
    # ax.plot(states[:, 0], states[:, 1], states[:, 2], label="True orbit")
    # ax.plot(estimated_states[:, 0], estimated_states[:, 1], estimated_states[:, 2], label="Estimated orbit")
    # ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], label="Landmarks")
    #
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")
    # ax.legend()
    # plt.show()


def load_brahe_data_files():
    brahe_directory = os.path.dirname(brahe.__file__)
    try:
        print("Updating Brahe data files. Might take a minute ...")
        brahe.utils.download_all_data(brahe_directory + "/data")
    except:
        pass  # One or the other files always errors out. Not a problem though


if __name__ == "__main__":
    np.random.seed(69420)
    load_brahe_data_files()
    test_od()

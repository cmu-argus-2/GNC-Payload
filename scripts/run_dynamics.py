"""
Generate the trajectory of the spacecraft using the dynamics model.

This script requires the following arguments:
    -f: Frequency of the spacecraft trajectory
    --mission_duration: Duration of the spacecraft mission for which we are generating the trajectory [s]
    --lat: Starting latitude of the spacecraft
    --lon: Starting longitude of the spacecraft
    --altitude: Starting altitude of the spacecraft [m]
    --name: Name of the experiment

The script will generate a ground truth trajectory of the spacecraft using the dynamics model and save this 
the experiment in the output directory.

- /output_dir
    - /experiment_name
        - trajectory_gt.npy
        - attitude_gt.npy
"""

import argparse
import os

import brahe
import numpy as np
import quaternion
from brahe.epoch import Epoch

from dynamics.orbital_dynamics import Dynamics
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from utils.config_utils import load_config
from utils.orbit_utils import get_sso_orbit_state


def generate_trajectory(args) -> None:
    """
    Generate the trajectory of a spacecraft using the dynamics model.
    :param args: The command line arguments.

    :return: None
    """
    config = load_config()
    config["solver"]["world_update_rate"] = args.f  # Hz
    config["mission"]["duration"] = args.mission_duration  # s

    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))

    data_manager = ODSimulationDataManager(starting_epoch, dt)

    initial_state = get_sso_orbit_state(
        starting_epoch, args.lat, args.lon, args.altitude, northwards=True
    )
    inital_rot = np.eye(3)
    data_manager.push_next_state(initial_state, inital_rot)

    w = np.array(args.angular_velocity)

    ground_truth_dynamics = Dynamics(
        config=config,
        use_drag=True,
        use_j2=True,
    )

    for _ in range(0, N - 1):
        state = data_manager.latest_state
        q = data_manager.latest_attitude

        next_state = ground_truth_dynamics.perturbed_f(
            x=state[0:6],
            dt=dt,
            epoch=data_manager.latest_epoch,
        )
        next_quat = quaternion.from_rotation_matrix(q) * quaternion.from_rotation_vector(w * dt)

        data_manager.push_next_state(next_state[0:6], quaternion.as_rotation_matrix(next_quat))

    # Save the trajectory and attitude data
    trajectory = data_manager.states
    attitude = data_manager.eci_Rs_body
    if not os.path.exists(f"output_dir/{args.name}"):
        print(f"Directory output_dir/{args.name} does not exist.")
        os.makedirs(f"output_dir/{args.name}")
        print(f"Created directory output_dir/{args.name}")
    np.save(f"output_dir/{args.name}/trajectory_gt.npy", trajectory)
    np.save(f"output_dir/{args.name}/attitude_gt.npy", attitude)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spacecraft trajectory")
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
        "--lat",
        type=float,
        default="0",
        help="Starting latitude of the spacecraft",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default="-73",
        help="Starting longitude of the spacecraft",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default="600_000",
        help="Starting altitude of the spacecraft [m]",
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
    args = parser.parse_args()

    generate_trajectory(args)

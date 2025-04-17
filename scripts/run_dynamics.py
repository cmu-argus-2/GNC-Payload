"""
Generate the trajectory of the spacecraft using the dynamics model.

This script requires the following arguments:
    --lat: Starting latitude of the spacecraft
    --lon: Starting longitude of the spacecraft
    --altitude: Starting altitude of the spacecraft [m]
    --name: Name of the experiment
    --start_date: Start date of the spacecraft mission [MJD]
    --northwards: Whether the spacecraft is moving northwards. If False, the spacecraft will move southwards.
    --angular_velocity: Angular velocity of the spacecraft [rad/s]
    --frequency: Frequency of the dynamics model [Hz]
    --duration: Duration of the spacecraft mission [s]

The script will generate a ground truth trajectory of the spacecraft using the dynamics model and save this 
the experiment in the output directory. It will store the spacecraft's position and velocity in trajectory_gt.npy,
the spacecraft's attitude in attitude_gt.npy and a boolean specifying whether the spacecraft is currently on the 
day (1) or night (0) side of the earth. We also store the arguments used to generate the trajectory in args.json.

- /output_dir
    - /experiment_name
        - trajectory_gt.npy
        - attitude_gt.npy
        - daytime_gt.npy
        - args.json
"""

import argparse
import json
import os

import brahe
import numpy as np
import quaternion
from brahe.epoch import Epoch
from tqdm import tqdm

from dynamics.orbital_dynamics import Dynamics
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from utils.config_utils import load_config, USER_CONFIG_PATH
from utils.orbit_utils import get_max_sso_latitude, get_sso_orbit_state, is_over_daytime


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate spacecraft trajectory")
    parser.add_argument(
        "--lat",
        type=float,
        # No default value. If not provided, it will be generated randomly within the SSO bounds.
        help="Starting latitude of the spacecraft",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=np.random.uniform(-180, 180),
        help="Starting longitude of the spacecraft",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=510e3 + np.random.uniform(-20e3, 20e3),
        help="Starting altitude of the spacecraft [m]",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="test",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--start_date",
        type=float,
        # Start date between 2024-01-01 and 2025-01-01 (leap year so 366 days)
        default=round(np.random.uniform(60310, 60676), 1),
        help="Start date of the spacecraft mission [MJD]",
    )
    parser.add_argument(
        "--northwards",
        type=bool,
        default=np.random.choice([True, False]),
        help="Whether the spacecraft is moving northwards. If False, the spacecraft will move southwards.",
    )
    parser.add_argument(
        "--angular_velocity",
        type=list,
        default=[0, 0, np.pi / 18],
        help="Angular velocity of the spacecraft [rad/s]",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=2,
        help="Frequency of the dynamics model [Hz]",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2700,
        help="Duration of the spacecraft mission [s]",
    )
    return parser.parse_args()


def generate_trajectory(args: argparse.Namespace) -> None:
    """
    Generate the trajectory of a spacecraft using the dynamics model.
    :param args: The command line arguments.

    :return: None
    """
    max_lat = get_max_sso_latitude(args.altitude)
    if args.lat is None:
        args.lat = np.random.uniform(-max_lat, max_lat)
    elif abs(args.lat) > max_lat:
        raise ValueError(
            f"Latitude must be between -{max_lat} and {max_lat} for an SSO orbit at an altitude of {args.altitude} m."
        )

    config = load_config()
    config["solver"]["world_update_rate"] = args.frequency  # Hz
    config["mission"]["duration"] = args.duration  # s
    config["mission"]["start_date"] = args.start_date

    dt = 1 / config["solver"]["world_update_rate"]

    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))
    daytime = []

    data_manager = ODSimulationDataManager(starting_epoch, dt)

    initial_state = get_sso_orbit_state(
        starting_epoch, args.lat, args.lon, args.altitude, northwards=args.northwards
    )
    initial_rot = np.eye(3)
    daytime.append(1 if is_over_daytime(starting_epoch, initial_state[0:3]) else 0)
    data_manager.push_next_state(initial_state, initial_rot)

    w = np.array(args.angular_velocity)

    ground_truth_dynamics = Dynamics(
        config=config,
        use_drag=True,
        use_j2=True,
        use_j34=False,
        use_moon_grav=False,
        use_sun_grav=False,
    )

    for _ in tqdm(range(0, N - 1)):
        state = data_manager.latest_state
        q = data_manager.latest_attitude

        next_state = ground_truth_dynamics.perturbed_f(
            x=state[0:6],
            dt=dt,
            epoch=data_manager.latest_epoch,
        )
        next_quat = quaternion.from_rotation_matrix(q) * quaternion.from_rotation_vector(w * dt)

        daytime.append(1 if is_over_daytime(data_manager.latest_epoch, state[0:3]) else 0)

        data_manager.push_next_state(next_state[0:6], quaternion.as_rotation_matrix(next_quat))

    # Save the trajectory and attitude data
    trajectory = data_manager.states
    attitude = data_manager.eci_Rs_body
    daytime = np.array(daytime)

    output_dir = os.path.join(load_config(USER_CONFIG_PATH)["output_directory"], args.name)
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
    np.save(f"{output_dir}/trajectory_gt.npy", trajectory)
    np.save(f"{output_dir}/attitude_gt.npy", attitude)
    np.save(f"{output_dir}/daytime_gt.npy", daytime)

    # Store args as json
    with open(f"{output_dir}/args.json", "w") as jsonfile:
        json.dump(args.__dict__, jsonfile, indent=4)


if __name__ == "__main__":
    generate_trajectory(parse_args())

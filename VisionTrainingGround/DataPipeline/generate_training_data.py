"""
Generate training data using the EarthImageSimulator for the specified MGRS regions.

This script will generate/overwrite the following contents in the training directory:
- /training_directory
  - /{region}
    - 00000.png
    - 00000_mgrs_regions.npy
    - 00000_lat_lon.npy
    - ...
"""

import argparse
import os
from functools import partial
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import cv2
import numpy as np
from brahe.constants import R_EARTH
from scipy.spatial.transform import Rotation

from image_simulation.earth_vis import EarthImageSimulator, GeoTIFFCache
from sensors.camera_model import CameraModelManager
from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.earth_utils import get_MGRS_grid, get_nadir_rotation, lat_lon_to_ecef
from utils.memory_aware_process_pool import MemoryAwareProcessPool

LAT_LON_OUTPUT_FILE_SUFFIX = "_lat_lon.npy"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for generating training data using the EarthImageSimulator.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate training data for the specified MGRS regions using the EarthImageSimulator."
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=load_config()["vision"]["salient_mgrs_region_ids"],
        help="MGRS regions to generate training data for.",
    )
    parser.add_argument(
        "--skip_regions",
        type=str,
        nargs="+",
        default=[],
        help="MGRS regions to skip. This takes precedence over --regions.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output directory if it exists."
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=int(0.8 * cpu_count()),
        help="Number of processes to use for generating training data in parallel.",
    )

    parser.add_argument(
        "--lat_lon_buffer",
        type=float,
        default=0.0,
        help="Extra buffer in lat/lon for each region, in degrees.",
    )
    parser.add_argument(
        "--num_images", type=int, default=1000, help="Number of images to generate per region."
    )
    parser.add_argument(
        "--nominal_altitude",
        type=float,
        default=510e3,
        help="Nominal altitude of the satellite, in m.",
    )
    parser.add_argument(
        "--altitude_variation", type=float, default=20e3, help="Variation in altitude, in m."
    )
    parser.add_argument(
        "--off_nadir_variation",
        type=float,
        default=10,
        help="Variation in off-nadir angle, in degrees.",
    )
    return parser.parse_args()


def setup_region_directory(region_dir: str, overwrite: bool) -> bool:
    """
    Create the region directory if it does not exist, or clear the output files that will be replaced if overwrite is
    True.

    :param region_dir: The path to the region directory.
    :param overwrite: Whether to overwrite the output files if they exist.
    :return: True if region_dir is now a directory that doesn't contain any of the output files that would be replaced,
             False otherwise.
    """
    if not os.path.exists(region_dir):
        os.makedirs(region_dir)
        return True

    if not os.path.isdir(region_dir):
        if not overwrite:
            return False
        os.remove(region_dir)
        os.makedirs(region_dir)
        return True

    conflicting_suffixes = [".png", LAT_LON_OUTPUT_FILE_SUFFIX]
    conflicting_file_names = [
        file_name
        for file_name in os.listdir(region_dir)
        if any(file_name.endswith(suffix) for suffix in conflicting_suffixes)
    ]
    if len(conflicting_file_names) == 0:
        return True

    if not overwrite:
        return False
    for conflicting_file_name in conflicting_file_names:
        os.remove(os.path.join(region_dir, conflicting_file_name))
    return True


def generate_training_image(
    region: str,
    file_prefix: str,
    lat_lon_buffer: float,
    nominal_altitude: float,
    altitude_variation: float,
    off_nadir_variation: float,
) -> None:
    """
    Generate a single training image using the EarthImageSimulator.

    This function generates 3 files:
    - A PNG image file containing the training image.
    - A .npy file with dtype=str containing the MGRS regions for each pixel.
    - A .npy file with containing the lat/lon coordinates for each pixel.

    :param region: The MGRS region to generate the training image for.
    :param file_prefix: The prefix for the output files.
    :param lat_lon_buffer: The extra buffer in possible lat/lon coordinates for the region.
    :param nominal_altitude: The nominal altitude of the satellite, in m.
    :param altitude_variation: The variation in altitude, in m. The actual altitude will be in the range
                               [nominal_altitude - altitude_variation, nominal_altitude + altitude_variation].
    :param off_nadir_variation: The maximum variation in off-nadir angle, in degrees.
    """
    min_lon, min_lat, max_lon, max_lat = get_MGRS_grid()[region]
    min_lon -= lat_lon_buffer
    min_lat -= lat_lon_buffer
    max_lon += lat_lon_buffer
    max_lat += lat_lon_buffer

    lat = np.random.uniform(min_lat, max_lat)
    lon = np.random.uniform(min_lon, max_lon)
    lat = np.clip(lat, -90, 90)
    lon = np.clip(lon, -180, 180)
    altitude = nominal_altitude + np.random.uniform(-altitude_variation, altitude_variation)

    ecef_position = lat_lon_to_ecef(np.array([lat, lon]))
    ecef_position /= (R_EARTH + altitude) / np.linalg.norm(ecef_position)
    ecef_velocity = np.array([0, 0, 1])

    camera_manager = CameraModelManager()
    perturbed_camera_R_nominal_camera = Rotation.from_euler(
        "ZX",
        [np.random.uniform(0, 360), np.random.uniform(0, off_nadir_variation)],
        degrees=True,
    ).as_matrix()
    nominal_body_R_nominal_camera = perturbed_body_R_perturbed_camera = camera_manager[
        "x+"
    ].body_R_camera
    ecef_R_nominal_body = get_nadir_rotation(np.concatenate((ecef_position, ecef_velocity)))
    ecef_R_perturbed_body = (
        ecef_R_nominal_body
        @ nominal_body_R_nominal_camera
        @ perturbed_camera_R_nominal_camera.T
        @ perturbed_body_R_perturbed_camera.T
    )

    image_simulator = EarthImageSimulator(GeoTIFFCache(max_cache_size=0))
    frame, lat_lon = image_simulator.simulate_image_for_training(
        ecef_position, ecef_R_perturbed_body, camera_manager["x+"]
    )

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    region_dir = os.path.join(training_dir, region)
    os.makedirs(region_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(region_dir, f"{file_prefix}.png"),
        cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR),
    )
    np.save(os.path.join(region_dir, f"{file_prefix}{LAT_LON_OUTPUT_FILE_SUFFIX}"), lat_lon)


def main() -> None:
    """
    Generate training data using the EarthImageSimulator.
    """
    args = parse_args()
    regions = list(set(args.regions) - set(args.skip_regions))
    total_images = len(regions) * args.num_images

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    for region in regions:
        region_dir: str = os.path.join(training_dir, region)
        if not setup_region_directory(region_dir, args.overwrite):
            print(
                f"Output directory {region_dir} could not be emptied. Set --overwrite to clear any existing data."
            )
            return

    func = partial(
        generate_training_image,
        lat_lon_buffer=args.lat_lon_buffer,
        nominal_altitude=args.nominal_altitude,
        altitude_variation=args.altitude_variation,
        off_nadir_variation=args.off_nadir_variation,
    )
    file_prefixes_generator = (f"{i:05d}" for i in range(args.num_images))
    if args.num_processes > 1:
        requests = list(product(regions, file_prefixes_generator))
        with MemoryAwareProcessPool(num_workers=args.num_processes) as pool:
            successful_requests, request_results = pool.map(func, requests)

        for request, success, result in zip(requests, successful_requests, request_results):
            if not success:
                print(f"Generation of training image for {request} failed with exception: {result}")
    else:
        for region, file_prefix in tqdm(
            product(regions, file_prefixes_generator), total=total_images
        ):
            func(region, file_prefix)


if __name__ == "__main__":
    main()

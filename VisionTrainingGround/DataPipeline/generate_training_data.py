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
from multiprocessing import cpu_count
from time import time

import cv2
import numpy as np
from brahe.constants import R_EARTH
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from image_simulation.earth_vis import EarthImageSimulator, GeoTIFFCache
from sensors.camera_model import CameraModel, CameraModelManager
from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.earth_utils import get_MGRS_grid, get_nadir_rotation, lat_lon_to_ecef
from utils.memory_aware_process_pool import MemoryAwareProcessPool

LAT_LON_OUTPUT_FILE_SUFFIX = "_lat_lon.npz"


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
        "--resume",
        action="store_true",
        help="Resume generating training data for all requests that failed in the previous run.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=int(0.5 * cpu_count()),
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


def setup_region_directory(region_dir: str, overwrite: bool, resume: bool) -> bool:
    """
    Set up the region directory for generating training data.

    This function will create the region directory if it does not exist.
    If overwrite is True, it will clear any output files that will be replaced.
    If resume is True, it will check for and remove any output files with partial or corrupted data.

    :param region_dir: The path to the region directory.
    :param overwrite: Whether to overwrite the output files if they exist. Cannot be True if resume is also True.
    :param resume: Whether to resume generating training data for all requests that failed in the previous run.
                   Cannot be True if overwrite is also True.
    :return: True if region_dir is now a directory that is ready for generating training data, False otherwise.
    """
    assert not (overwrite and resume), "Overwrite and resume cannot both be True."

    if not os.path.exists(region_dir):
        os.makedirs(region_dir)
        return True

    if not os.path.isdir(region_dir):
        if not overwrite:
            return False
        os.remove(region_dir)
        os.makedirs(region_dir)
        return True

    file_names = os.listdir(region_dir)
    existing_image_file_names = [
        file_name for file_name in file_names if file_name.endswith(".png")
    ]
    existing_lat_lon_file_names = [
        file_name for file_name in file_names if file_name.endswith(LAT_LON_OUTPUT_FILE_SUFFIX)
    ]
    if len(existing_image_file_names) == 0 and len(existing_lat_lon_file_names) == 0:
        return True

    if overwrite:
        for file_name in existing_image_file_names + existing_lat_lon_file_names:
            os.remove(os.path.join(region_dir, file_name))
        return True

    def are_files_corrupted(common_file_name_: str) -> bool:
        """
        Check if the image and lat/lon files with the given common file name are corrupted.

        :param common_file_name_: The common file name to check.
        :return: True if the files are corrupted, False otherwise.
        """
        try:
            img = cv2.imread(os.path.join(region_dir, f"{common_file_name_}.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lat_lon = np.load(
                os.path.join(region_dir, f"{common_file_name_}{LAT_LON_OUTPUT_FILE_SUFFIX}")
            )["lat_lon"]
            if img is None or lat_lon is None:
                return True

            if img.shape != CameraModel.OUTPUT_SHAPE or img.dtype != CameraModel.DTYPE:
                return True
            if lat_lon.shape != (CameraModel.RESOLUTION, 2) or not np.issubdtype(
                lat_lon.dtype, np.floating
            ):
                return True
        except Exception:
            return True
        return False

    if resume:
        existing_image_file_names = set([file_name[:-4] for file_name in existing_image_file_names])
        existing_lat_lon_file_names = set(
            [
                file_name[: -len(LAT_LON_OUTPUT_FILE_SUFFIX)]
                for file_name in existing_lat_lon_file_names
            ]
        )

        # delete files without a counterpart
        for file_name in existing_image_file_names - existing_lat_lon_file_names:
            os.remove(os.path.join(region_dir, f"{file_name}.png"))
        for file_name in existing_lat_lon_file_names - existing_image_file_names:
            os.remove(os.path.join(region_dir, f"{file_name}{LAT_LON_OUTPUT_FILE_SUFFIX}"))

        for common_file_name in existing_image_file_names & existing_lat_lon_file_names:
            if are_files_corrupted(common_file_name):
                os.remove(os.path.join(region_dir, f"{common_file_name}.png"))
                os.remove(
                    os.path.join(region_dir, f"{common_file_name}{LAT_LON_OUTPUT_FILE_SUFFIX}")
                )

        return True

    # there are existing files but overwrite and resume are both False
    return False


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
    np.savez_compressed(
        os.path.join(region_dir, f"{file_prefix}{LAT_LON_OUTPUT_FILE_SUFFIX}"), lat_lon=lat_lon
    )


def main() -> None:
    """
    Generate training data using the EarthImageSimulator.
    """
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("Cannot use --overwrite and --resume at the same time.")

    regions = list(set(args.regions) - set(args.skip_regions))
    total_images = len(regions) * args.num_images
    if total_images == 0:
        print("No training images to generate.")
        return

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    for region in tqdm(regions, desc="Setting up region directories"):
        region_dir: str = os.path.join(training_dir, region)
        if not setup_region_directory(region_dir, args.overwrite, args.resume):
            print(
                f"Output directory {region_dir} could not be emptied. Set --overwrite to clear any existing data."
            )
            return

    file_prefixes_generator = (f"{i:05d}" for i in range(args.num_images))
    requests_generator = product(regions, file_prefixes_generator)
    if args.resume:
        requests_generator = (
            (region, file_prefix)
            for region, file_prefix in requests_generator
            if not os.path.exists(os.path.join(training_dir, region, f"{file_prefix}.png"))
        )

    func = partial(
        generate_training_image,
        lat_lon_buffer=args.lat_lon_buffer,
        nominal_altitude=args.nominal_altitude,
        altitude_variation=args.altitude_variation,
        off_nadir_variation=args.off_nadir_variation,
    )
    if args.num_processes > 1:
        requests = list(requests_generator)
        log_file_path = os.path.join(training_dir, f"training_data_generation_log_{time()}.csv")
        with MemoryAwareProcessPool(num_workers=args.num_processes) as pool:
            successful_requests, request_results = pool.map(
                func, requests, output_log_path=log_file_path
            )

        for request, success, result in zip(requests, successful_requests, request_results):
            if not success:
                print(f"Generation of training image for {request} failed with exception: {result}")
    else:
        for region, file_prefix in tqdm(requests_generator, total=total_images):
            func(region, file_prefix)


if __name__ == "__main__":
    main()

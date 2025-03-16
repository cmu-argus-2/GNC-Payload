"""
Generate training data using the EarthImageSimulator.
"""

import argparse
import os
from shutil import rmtree

import cv2
import numpy as np
from brahe.constants import R_EARTH
from scipy.spatial.transform import Rotation

from image_simulation.earth_vis import EarthImageSimulator, GeoTIFFCache
from sensors.camera_model import CameraModelManager
from utils.config_utils import load_config, USER_CONFIG_PATH
from utils.earth_utils import get_MGRS_grid, get_nadir_rotation, lat_lon_to_ecef

MGRS_REGIONS_OUTPUT_FILE_SUFFIX = "_mgrs_regions.npy"
LAT_LON_OUTPUT_FILE_SUFFIX = "_lat_lon.npy"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for generating training data using the EarthImageSimulator.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate training data using the EarthImageSimulator."
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=load_config()["vision"]["salient_mgrs_region_ids"],
        help="MGRS regions to generate training data for.",
    )
    parser.add_argument(
        "--skip_regions", type=str, nargs="+", default=[], help="MGRS regions to skip."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output directory if it exists."
    )
    parser.add_argument(
        "--lat_lon_buffer", type=float, default=0.0, help="Extra buffer in lat/lon for each region."
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
    Create the region directory if it does not exist, or clear it if it does and overwrite is True.

    :param region_dir: The path to the region directory.
    :param overwrite: Whether to overwrite the directory if it exists.
    :return: True if region_dir is now an empty directory, False otherwise.
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

    if len(os.listdir(region_dir)) == 0:
        return True

    if not overwrite:
        return False
    rmtree(region_dir)
    os.makedirs(region_dir)
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
    altitude = nominal_altitude + np.random.uniform(
        -altitude_variation, altitude_variation
    )

    ecef_position = lat_lon_to_ecef(lat, lon)
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
    frame, mgrs_regions, lat_lon = image_simulator.simulate_image_for_training(
        ecef_position, ecef_R_perturbed_body, camera_manager["x+"]
    )

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    region_dir = os.path.join(training_dir, region)
    os.makedirs(region_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(region_dir, f"{file_prefix}.png"),
        cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR),
    )
    np.save(
        os.path.join(region_dir, f"{file_prefix}{MGRS_REGIONS_OUTPUT_FILE_SUFFIX}"),
        mgrs_regions,
    )
    np.save(os.path.join(region_dir, f"{file_prefix}{LAT_LON_OUTPUT_FILE_SUFFIX}"), lat_lon)


def main() -> None:
    """
    Generate training data using the EarthImageSimulator.
    """
    args = parse_args()
    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    regions = list(set(args.regions) - set(args.skip_regions))

    for region in regions:
        region_dir: str = os.path.join(training_dir, region)
        if not setup_region_directory(region_dir, args.overwrite):
            print(
                f"Output directory {args.output_dir} could not be emptied. Set --overwrite to clear any existing data."
            )
            return

        for i in range(args.num_images):
            file_prefix = f"{i:05d}"

            generate_training_image(
                region,
                file_prefix,
                args.lat_lon_buffer,
                args.nominal_altitude,
                args.altitude_variation,
                args.off_nadir_variation,
            )


if __name__ == "__main__":
    main()

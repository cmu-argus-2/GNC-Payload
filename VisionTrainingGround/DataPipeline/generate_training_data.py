"""
Generate training data using the EarthImageSimulator.
"""

import argparse
import os
import pickle
from shutil import rmtree

import numpy as np
from brahe.constants import R_EARTH
from scipy.spatial.transform import Rotation

from image_simulation.earth_vis import EarthImageSimulator, GeoTIFFCache
from sensors.camera_model import CameraModelManager
from utils.config_utils import load_config
from utils.earth_utils import get_MGRS_grid, get_nadir_rotation, lat_lon_to_ecef


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for generating training data using the EarthImageSimulator.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate training data using the EarthImageSimulator."
    )

    parser.add_argument(
        "--geotiff_folder",
        type=str,
        default=GeoTIFFCache.get_default_geotiff_folder(),
        help="Path to the folder containing GeoTIFF files for each region.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save the training data."
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output directory if it exists."
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


def setup_output_directory(output_dir: str, overwrite: bool) -> bool:
    """
    Create the output directory if it does not exist, or clear it if it does and overwrite is True.

    :param output_dir: The path to the output directory.
    :param overwrite: Whether to overwrite the directory if it exists.
    :return: True if output_dir is now an empty directory, False otherwise.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return True

    if not os.path.isdir(output_dir):
        if not overwrite:
            return False
        os.remove(output_dir)
        os.makedirs(output_dir)
        return True

    if len(os.listdir(output_dir)) == 0:
        return True

    if not overwrite:
        return False
    rmtree(output_dir)
    os.makedirs(output_dir)
    return True


def main() -> None:
    """
    Generate training data using the EarthImageSimulator.
    """
    args = parse_args()
    if not setup_output_directory(args.output_dir, args.overwrite):
        print(
            f"Output directory {args.output_dir} could not be emptied. Set --overwrite to clear any existing data."
        )
        return

    config = load_config()
    image_simulator = EarthImageSimulator(
        GeoTIFFCache(geotiff_folder=args.geotiff_folder, max_cache_size=0)
    )
    camera_manager = CameraModelManager()
    grid = get_MGRS_grid()
    ecef_velocity = np.array([0, 0, 1])

    for region in config["vision"]["salient_mgrs_region_ids"]:
        region_dir = os.path.join(args.output_dir, region)
        os.makedirs(region_dir)

        min_lon, min_lat, max_lon, max_lat = grid[region]
        for i in range(args.num_images):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            altitude = args.nominal_altitude + np.random.uniform(
                -args.altitude_variation, args.altitude_variation
            )

            ecef_position = lat_lon_to_ecef(lat, lon)
            ecef_position /= (R_EARTH + altitude) / np.linalg.norm(ecef_position)

            perturbed_camera_R_nominal_camera = Rotation.from_euler(
                "ZX",
                [np.random.uniform(0, 360), np.random.uniform(0, args.off_nadir_variation)],
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

            frame, mgrs_regions, lat_lon = image_simulator.simulate_image_for_training(
                ecef_position, ecef_R_perturbed_body, camera_manager["x+"]
            )

            # TODO: integrate with RC and LD training scripts
            pickle_path = os.path.join(region_dir, f"{i:05d}.pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(
                    {
                        "image": frame.image,
                        "mgrs_regions": mgrs_regions,
                        "lat_lon": lat_lon,
                    },
                    f,
                )


if __name__ == "__main__":
    main()

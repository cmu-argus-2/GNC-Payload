"""
Generate training data using the EarthImageSimulator for the specified MGRS regions.

This script will generate/overwrite the following contents in the training directory:
- /training_directory
  - /{region}
    - 00000.png
    - 00000_lat_lon.npy
    - ...
"""

import argparse
from dataclasses import dataclass
import os
from functools import partial
from itertools import product
from multiprocessing import cpu_count, Pool
from time import perf_counter, time
from typing import ClassVar, Generator, Tuple

import cv2
import numpy as np
from brahe.constants import R_EARTH
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from image_simulation.earth_vis import EarthImageSimulator, GeoTIFFCache
from sensors.camera_model import CameraModel, CameraModelManager
from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.earth_utils import get_MGRS_grid, get_nadir_rotation, lat_lon_to_ecef
from utils.function_utils import unpack_and_call


@dataclass
class GeotaggedImage:
    """
    A class representing an image alongside lat/lon coordinates for each pixel.

    Attributes:
        image: A numpy array of shape CameraModel.RESOLUTION + (3,) containing the RGB image.
        lat_lon: A numpy array of shape CameraModel.RESOLUTION + (2,) containing the latitudes and longitudes for each
                 pixel, or np.nan if the pixel does not intersect the Earth.
    """

    IMAGE_SUFFIX: ClassVar[str] = ".png"
    LAT_LON_SUFFIX: ClassVar[str] = "_lat_lon.npz"

    image: np.ndarray
    lat_lon: np.ndarray

    def assert_invariants(self) -> None:
        """
        :raises AssertionError: If the image or lat/lon coordinates are invalid.
        """
        assert self.image is not None and self.lat_lon is not None
        assert self.image.shape == CameraModel.OUTPUT_SHAPE
        assert self.image.dtype == CameraModel.DTYPE
        assert self.lat_lon.shape == CameraModel.RESOLUTION + (2,)
        assert np.issubdtype(self.lat_lon.dtype, np.floating)

    def save(self, region: str, file_prefix: str) -> None:
        """
        Save the image and lat/lon coordinates to the specified region and file prefix.

        :param region: The MGRS region.
        :param file_prefix: The prefix for the output files.
        """
        self.assert_invariants()

        training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
        region_dir = os.path.join(training_dir, region)
        os.makedirs(region_dir, exist_ok=True)

        cv2.imwrite(
            os.path.join(region_dir, f"{file_prefix}{GeotaggedImage.IMAGE_SUFFIX}"),
            cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
        )
        np.savez_compressed(
            os.path.join(region_dir, f"{file_prefix}{GeotaggedImage.LAT_LON_SUFFIX}"),
            lat_lon=self.lat_lon,
        )

    @staticmethod
    def load(region: str, file_prefix: str) -> "GeotaggedImage":
        """
        Load the image and lat/lon coordinates from the specified region and file prefix.

        :param region: The MGRS region.
        :param file_prefix: The prefix for the output files.
        :return: A GeotaggedImage object containing the loaded image and lat/lon coordinates.
        """
        training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
        region_dir = os.path.join(training_dir, region)

        image = cv2.imread(os.path.join(region_dir, f"{file_prefix}{GeotaggedImage.IMAGE_SUFFIX}"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        lat_lon = np.load(
            os.path.join(region_dir, f"{file_prefix}{GeotaggedImage.LAT_LON_SUFFIX}")
        )["lat_lon"]

        geotagged_image = GeotaggedImage(image, lat_lon)
        geotagged_image.assert_invariants()
        return geotagged_image


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
        default=0.25,
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


def setup_region_directory(
    region_dir: str, overwrite: bool, resume: bool, check_corrupted: bool = False
) -> bool:
    """
    Set up the region directory for generating training data.

    This function will create the region directory if it does not exist.
    If overwrite is True, it will clear any output files that will be replaced.
    If resume is True, it will check for and remove any output files with partial or corrupted data.

    :param region_dir: The path to the region directory.
    :param overwrite: Whether to overwrite the output files if they exist. Cannot be True if resume is also True.
    :param resume: Whether to resume generating training data for all requests that failed in the previous run.
                   Cannot be True if overwrite is also True.
    :param check_corrupted: Whether to check for corrupted files in the region directory before resuming. This can only
                            be True if resume is also True. This is very expensive since it loads all existing image and
                            lat/lon files into memory. You're probably better off just crossing your fingers and hoping
                            for the best.
    :return: True if region_dir is now a directory that is ready for generating training data, False otherwise.
    """
    assert not (overwrite and resume), "Overwrite and resume cannot both be True."
    assert not (
        check_corrupted and not resume
    ), "Check corrupted files cannot be True if resume is False."

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
        file_name for file_name in file_names if file_name.endswith(GeotaggedImage.IMAGE_SUFFIX)
    ]
    existing_lat_lon_file_names = [
        file_name for file_name in file_names if file_name.endswith(GeotaggedImage.LAT_LON_SUFFIX)
    ]
    if len(existing_image_file_names) == 0 and len(existing_lat_lon_file_names) == 0:
        return True

    if overwrite:
        for file_name in existing_image_file_names + existing_lat_lon_file_names:
            os.remove(os.path.join(region_dir, file_name))
        return True

    if resume:
        existing_image_file_names = set(
            [
                file_name[: -len(GeotaggedImage.IMAGE_SUFFIX)]
                for file_name in existing_image_file_names
            ]
        )
        existing_lat_lon_file_names = set(
            [
                file_name[: -len(GeotaggedImage.LAT_LON_SUFFIX)]
                for file_name in existing_lat_lon_file_names
            ]
        )

        # delete files without a counterpart
        for file_name in existing_image_file_names - existing_lat_lon_file_names:
            os.remove(os.path.join(region_dir, f"{file_name}{GeotaggedImage.IMAGE_SUFFIX}"))
        for file_name in existing_lat_lon_file_names - existing_image_file_names:
            os.remove(os.path.join(region_dir, f"{file_name}{GeotaggedImage.LAT_LON_SUFFIX}"))

        if check_corrupted:
            for common_file_name in tqdm(
                existing_image_file_names & existing_lat_lon_file_names,
                desc=f"Checking for corrupted files in {region_dir}",
            ):
                try:
                    GeotaggedImage.load(region_dir, common_file_name)
                except Exception:
                    os.remove(os.path.join(region_dir, f"{common_file_name}{GeotaggedImage.IMAGE_SUFFIX}"))
                    os.remove(
                        os.path.join(region_dir, f"{common_file_name}{GeotaggedImage.LAT_LON_SUFFIX}")
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
    # Without this the seed may be inherited from the calling process, leading to duplicate images
    rng = np.random.default_rng(np.random.SeedSequence(int(time() * 1e6) ^ os.getpid()))

    min_lon, min_lat, max_lon, max_lat = get_MGRS_grid()[region]
    min_lon -= lat_lon_buffer
    min_lat -= lat_lon_buffer
    max_lon += lat_lon_buffer
    max_lat += lat_lon_buffer

    lat = rng.uniform(min_lat, max_lat)
    lon = rng.uniform(min_lon, max_lon)
    lat = np.clip(lat, -90, 90)
    lon = np.clip(lon, -180, 180)
    altitude = nominal_altitude + rng.uniform(-altitude_variation, altitude_variation)

    ecef_position = lat_lon_to_ecef(np.array([lat, lon]))
    ecef_position *= (R_EARTH + altitude) / np.linalg.norm(ecef_position)
    ecef_velocity = np.array([0, 0, 1])

    camera_manager = CameraModelManager()
    perturbed_camera_R_nominal_camera = Rotation.from_euler(
        "ZXZ",
        [rng.uniform(0, 360), rng.uniform(0, off_nadir_variation), rng.uniform(0, 360)],
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

    geotagged_image = GeotaggedImage(frame.image, lat_lon)
    geotagged_image.save(region, file_prefix)


def main() -> None:
    """
    Generate training data using the EarthImageSimulator.
    """
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("Cannot use --overwrite and --resume at the same time.")
    regions = sorted(set(args.regions) - set(args.skip_regions))

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    for region in tqdm(regions, desc="Setting up region directories"):
        region_dir: str = os.path.join(training_dir, region)
        if not setup_region_directory(region_dir, args.overwrite, args.resume):
            print(
                f"Output directory {region_dir} could not be emptied. Set --overwrite to clear any existing data."
            )
            return

    def get_requests_generator() -> Generator[Tuple[str, str], None, None]:
        """
        :return: A generator that yields tuples of (region, file_prefix) for each image to be generated.
        """
        file_prefixes_generator = (f"{i:05d}" for i in range(args.num_images))
        requests_generator = product(regions, file_prefixes_generator)

        if args.resume:
            requests_generator = (
                (region_, file_prefix_)
                for region_, file_prefix_ in requests_generator
                if not os.path.exists(os.path.join(training_dir, region_, f"{file_prefix_}.png"))
            )
        return requests_generator

    total_images = (
        sum(1 for _ in get_requests_generator()) if args.resume else len(regions) * args.num_images
    )
    if total_images == 0:
        print("No training images to generate.")
        return
    args.num_processes = min(args.num_processes, total_images)

    func = partial(
        generate_training_image,
        lat_lon_buffer=args.lat_lon_buffer,
        nominal_altitude=args.nominal_altitude,
        altitude_variation=args.altitude_variation,
        off_nadir_variation=args.off_nadir_variation,
    )
    if args.num_processes > 1:
        log_file_path = os.path.join(training_dir, f"training_data_generation_log_{time()}.csv")
        with Pool(args.num_processes) as pool:
            results_iterator = tqdm(
                pool.imap_unordered(
                    partial(
                        unpack_and_call,
                        func,
                    ),
                    get_requests_generator(),
                    chunksize=1,
                ),
                total=total_images,
                desc="Generating images",
            )
            start_time = perf_counter()
            with open(log_file_path, "w") as log_file:
                log_file.write("Elapsed Time (s), Number of Images Generated\n")
                for i, _ in enumerate(results_iterator):
                    log_file.write(f"{perf_counter() - start_time}, {i + 1}\n")
    else:
        for region, file_prefix in tqdm(
            get_requests_generator(), total=total_images, desc="Generating images"
        ):
            func(region, file_prefix)


if __name__ == "__main__":
    main()

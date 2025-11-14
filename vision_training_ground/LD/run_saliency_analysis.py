"""
Run saliency analysis for the specified MGRS regions.

This script expects to find the following contents in the training directory:
- /training_directory
  - /{region}
    - 00000.png
    - 00000_lat_lon.npz
    - ...

This script will generate/overwrite the following contents in the training directory:
- /training_directory
  - /{region}
    - saliency_map.tif
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Iterable, List

import cv2
import numpy as np
from affine import Affine
from tqdm import tqdm

from image_simulation.earth_vis import GeoTIFFData
from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.earth_utils import get_mgrs_region_dimensions
from vision_training_ground.DataPipeline.generate_training_data import GeotaggedImage

SALIENCY_MAP_FILE_NAME = "saliency_map.tif"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run saliency analysis for the specified MGRS regions."
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=load_config()["vision"]["salient_mgrs_region_ids"],
        help="MGRS regions to run saliency analysis for.",
    )
    parser.add_argument(
        "--skip_regions",
        type=str,
        nargs="+",
        default=[],
        help="MGRS regions to skip. This takes precedence over --regions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output file if it exists.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=int(0.8 * cpu_count()),
        help="Number of processes to use for running saliency analysis in parallel across the specified regions.",
    )

    parser.add_argument(
        "--gsd",
        type=float,
        default=50.0,
        help="The ground sample distance to use for the saliency map.",
    )
    return parser.parse_args()


def get_common_file_name_prefixes(input_dir: str, ignore_names: Iterable[str] = ()) -> List[str]:
    """
    Get the common file name prefixes for PNG and .npy lat/lon files in the input directory.
    A warning is printed if there are PNG files without corresponding lat/lon files, or vice versa.

    :param input_dir: The directory containing the PNGs and .npy files.
    :param ignore_names: An iterable of file names to ignore.
    :return: A list of common file name prefixes.
    """
    file_names = sorted(os.listdir(input_dir))
    image_file_prefixes = {
        name[: -len(GeotaggedImage.IMAGE_SUFFIX)]
        for name in file_names
        if name.endswith(GeotaggedImage.IMAGE_SUFFIX) and name not in ignore_names
    }
    lat_lon_file_prefixes = {
        name[: -len(GeotaggedImage.LAT_LON_SUFFIX)]
        for name in file_names
        if name.endswith(GeotaggedImage.LAT_LON_SUFFIX) and name not in ignore_names
    }
    common_file_prefixes = list(image_file_prefixes & lat_lon_file_prefixes)

    if len(image_file_prefixes) != len(common_file_prefixes):
        print(
            f"Warning: Some PNG files do not have corresponding lat/lon files: "
            f"{list(image_file_prefixes - lat_lon_file_prefixes)}"
        )
    if len(lat_lon_file_prefixes) != len(common_file_prefixes):
        print(
            f"Warning: Some lat/lon files do not have corresponding PNG files: "
            f"{list(lat_lon_file_prefixes - image_file_prefixes)}"
        )
    return common_file_prefixes


# pylint: disable=too-many-locals
def generate_saliency_map(
    region_dir: str, output_file: str, gsd: float, region_id: str
) -> GeoTIFFData:
    """
    Generate a saliency map for a MGRS region from a directory of PNGs and .npy files.

    :param region_dir: The directory containing the PNGs and .npy files for the region to use to generate the saliency
                       map.
    :param output_file: The file to save the saliency map to.
    :param gsd: The ground sample distance to use for the saliency map.
    :param region_id: The MGRS region to generate the saliency map for.
    :return: The saliency map as a GeoTIFFData object.
    """
    file_prefixes = get_common_file_name_prefixes(region_dir)
    if len(file_prefixes) == 0:
        raise ValueError("No matching PNG and lat/lon files found.")

    region_height, region_top_width, region_bottom_width = get_mgrs_region_dimensions(region_id)
    height = int(np.ceil(region_height / gsd))
    width = int(np.ceil(np.maximum(region_top_width, region_bottom_width) / gsd))

    region_saliency_map = GeoTIFFData(
        image_path=output_file,
        image_data=np.zeros((height, width, 1), dtype=np.float32),
        transform=Affine.identity(),
    )
    region_saliency_map.remap_to_mgrs_region(region_id)
    region_saliency_map_counts = np.zeros((height, width), dtype=int)
    saliency_computer: cv2.saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    for file_prefix in tqdm(file_prefixes, desc=f"Processing images for region {region_id}"):
        try:
            geotagged_image = GeotaggedImage.load(region_id, file_prefix)
        except Exception:
            print(
                f"Warning: Failed to load image for: {region_id=}, {file_prefix=}. Skipping this for saliency analysis."
            )
            continue

        success, img_saliency_map = saliency_computer.computeSaliency(
            cv2.cvtColor(geotagged_image.image, cv2.COLOR_RGB2BGR)
        )

        if not success:
            print(f"Warning: Failed to compute saliency map for: {region_id=}, {file_prefix=}")
            continue

        assert (
            img_saliency_map.dtype == region_saliency_map.dtype
        ), f"Expected saliency map to have dtype {region_saliency_map.dtype}, but got {img_saliency_map.dtype}."

        us, vs, valid_mask = region_saliency_map.get_pixel_coordinates(geotagged_image.lat_lon)

        # cannot use += because it won't work with repeated indices
        np.add.at(
            region_saliency_map.image_data,
            (vs[valid_mask], us[valid_mask], 0),
            img_saliency_map[valid_mask],
        )
        np.add.at(region_saliency_map_counts, (vs[valid_mask], us[valid_mask]), 1)

    nonzero_mask = region_saliency_map_counts > 0
    region_saliency_map.image_data[nonzero_mask, 0] /= region_saliency_map_counts[nonzero_mask]
    return region_saliency_map


def run_saliency_analysis_for_region(region: str, overwrite: bool, gsd: float) -> None:
    """
    Run the saliency analysis for a single region.

    :param region: The MGRS region to generate the saliency map for.
    :param overwrite: Whether to overwrite the output file if it exists.
    :param gsd: The ground sample distance to use for the saliency map.
    """
    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    region_dir = os.path.join(training_dir, region)
    saliency_map_file = os.path.join(region_dir, SALIENCY_MAP_FILE_NAME)
    if os.path.exists(saliency_map_file):
        if not overwrite:
            print(
                f"Output file {saliency_map_file} already exists and --overwrite is not set. Skipping."
            )
            return
        os.remove(saliency_map_file)

    print(f"Running saliency analysis for region {region}...")

    saliency_map = generate_saliency_map(region_dir, saliency_map_file, gsd, region)
    saliency_map.save()


def main() -> None:
    """
    Script entry point.
    """
    args = parse_args()
    regions = sorted(set(args.regions) - set(args.skip_regions))
    args.num_processes = min(args.num_processes, len(regions))

    func = partial(
        run_saliency_analysis_for_region,
        overwrite=args.overwrite,
        gsd=args.gsd,
    )
    if args.num_processes > 1:
        with Pool(args.num_processes) as pool:
            pool.map(func, regions)
    else:
        # we don't care about the results, just exhaust the iterator
        list(map(func, regions))


if __name__ == "__main__":
    main()

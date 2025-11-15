"""
Counts the number of occurrences of each MGRS region in each image.

This script expects to find the following contents in the training directory:
- /training_directory
  - /{region}
    - 00000_lat_lon.npz
    - ...

This scipy will generate/overwrite the following contents in the training directory:
- /training_directory
  - /{region}
    - 00000_mgrs_counts.json
    - ...
"""

import argparse
import json
import os
from functools import partial
from itertools import starmap
from multiprocessing import Pool, cpu_count
from typing import Generator, Tuple

import numpy as np
from tqdm import tqdm

from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.earth_utils import calculate_mgrs_zones
from utils.function_utils import unpack_and_call
from vision_training_ground.DataPipeline.generate_training_data import GeotaggedImage

MGRS_COUNTS_SUFFIX = "_mgrs_counts.json"


# pylint: disable=R0801
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for counting MGRS regions.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Counts the number of occurrences of each MGRS region in each image."
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=load_config()["vision"]["salient_mgrs_region_ids"],
        help="MGRS regions to count occurrences for.",
    )
    parser.add_argument(
        "--skip_regions",
        type=str,
        nargs="+",
        default=[],
        help="MGRS regions to skip. This takes precedence over --regions.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file if it exists."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume counting MGRS regions for all requests that failed in the previous run.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=int(0.5 * cpu_count()),
        help="Number of processes to use for counting MGRS regions in parallel.",
    )

    return parser.parse_args()


def setup_region_dir(region_id: str, overwrite: bool, resume: bool) -> bool:
    """
    Set up the region directory for counting MGRS regions.

    :param region_id: The region ID to set up.
    :param overwrite: Whether to overwrite existing files. Cannot be True if resume is also True.
    :param resume: Whether to resume counting. Cannot be True if overwrite is also True.
    :return: True if the region directory is set up successfully, False otherwise.
    """
    assert not (resume and overwrite), "resume and overwrite cannot be used together."

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    region_dir = os.path.join(training_dir, region_id)
    existing_files = [
        file_name for file_name in os.listdir(region_dir) if file_name.endswith(MGRS_COUNTS_SUFFIX)
    ]

    if len(existing_files) == 0:
        return True

    if overwrite:
        for file_name in existing_files:
            os.remove(os.path.join(region_dir, file_name))
        return True

    return True


def count_mgrs_regions(region_id: str, file_prefix: str) -> None:
    """
    Count the number of occurrences of each MGRS region in a geotagged image.

    :param region_id: The region ID to process.
    :param file_prefix: The file prefix for the geotagged image.
    """
    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    lat_lon_path = os.path.join(
        training_dir, region_id, f"{file_prefix}{GeotaggedImage.LAT_LON_SUFFIX}"
    )
    # pylint: disable=E1129
    with np.load(lat_lon_path) as data:
        lat_lon = data["lat_lon"]

    mgrs_regions = calculate_mgrs_zones(lat_lon)
    present_regions = np.unique(mgrs_regions)
    counts = {
        str(region, encoding="ascii"): int(np.sum(mgrs_regions == region))
        for region in present_regions  # pylint: disable=E1133
    }

    output_path = os.path.join(training_dir, region_id, f"{file_prefix}{MGRS_COUNTS_SUFFIX}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=4)


def main() -> None:
    """
    Script entry point.
    """
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("Cannot use --overwrite and --resume at the same time.")
    regions = sorted(set(args.regions) - set(args.skip_regions))
    args.num_processes = min(args.num_processes, len(regions))

    for region in tqdm(regions, desc="Setting up region directories"):
        if not setup_region_dir(region, args.overwrite, args.resume):
            print(
                f"Output files for {region} already exist. Set --overwrite to clear any existing data."
            )
            return

    def get_requests_generator() -> Generator[Tuple[str, str], None, None]:
        """
        :return: A generator that yields tuples of (region, file_prefix) for each request.
        """
        training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
        for region in regions:
            region_dir = os.path.join(training_dir, region)

            file_prefixes_generator = (
                file_name[: -len(GeotaggedImage.LAT_LON_SUFFIX)]
                for file_name in sorted(os.listdir(region_dir))
                if file_name.endswith(GeotaggedImage.LAT_LON_SUFFIX)
            )
            if args.resume:
                file_prefixes_generator = (
                    file_prefix
                    for file_prefix in file_prefixes_generator
                    if not os.path.exists(
                        os.path.join(region_dir, f"{file_prefix}{MGRS_COUNTS_SUFFIX}")
                    )
                )

            for file_prefix in file_prefixes_generator:
                yield region, file_prefix

    total_requests = sum(1 for _ in get_requests_generator())
    if args.num_processes > 1:
        with Pool(args.num_processes) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        partial(unpack_and_call, count_mgrs_regions),
                        get_requests_generator(),
                        chunksize=1,
                    ),
                    total=total_requests,
                    desc="Counting MGRS regions",
                )
            )
    else:
        list(
            tqdm(
                starmap(count_mgrs_regions, get_requests_generator()),
                desc="Counting MGRS regions",
                total=total_requests,
            )
        )


if __name__ == "__main__":
    main()

"""
Run saliency analysis for the specified MGRS regions.

This script expects to find the following contents in the training directory:
- /training_directory
  - /{region}
    - 00000.png
    - 00000_lat_lon.npy
    - ...

This script will generate/overwrite the following contents in the training directory:
- /training_directory
  - /{region}
    - saliency_map.tif
    - bounding_boxes.csv
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List

import cv2
import numpy as np
from affine import Affine
from brahe.constants import R_EARTH
from scipy.ndimage import uniform_filter
from tqdm import tqdm

from image_simulation.earth_vis import GeoTIFFData
from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.earth_utils import get_mgrs_region_dimensions
from vision_inference.landmark_detector import LandmarkDetector
from VisionTrainingGround.DataPipeline.generate_training_data import GeotaggedImage

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
    parser.add_argument(
        "--bounding_box_size",
        type=int,
        default=7200,
        help="The side length of the bounding boxes to find in the saliency map, in meters.",
    )
    parser.add_argument(
        "--num_boxes",
        type=int,
        default=50,
        help="The number of top saliency bounding boxes to identify.",
    )
    return parser.parse_args()


def get_common_file_name_prefixes(input_dir: str) -> List[str]:
    """
    Get the common file name prefixes for PNG and .npy lat/lon files in the input directory.
    A warning is printed if there are PNG files without corresponding lat/lon files, or vice versa.

    :param input_dir: The directory containing the PNGs and .npy files.
    :return: A list of common file name prefixes.
    """
    file_names = sorted(os.listdir(input_dir))
    image_file_prefixes = {
        name[: -len(GeotaggedImage.IMAGE_SUFFIX)]
        for name in file_names
        if name.endswith(GeotaggedImage.IMAGE_SUFFIX)
    }
    lat_lon_file_prefixes = {
        name[: -len(GeotaggedImage.LAT_LON_SUFFIX)]
        for name in file_names
        if name.endswith(GeotaggedImage.LAT_LON_SUFFIX)
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
    saliency_computer = cv2.saliency.StaticSaliencyFineGrained_create()

    for file_prefix in tqdm(file_prefixes, desc=f"Processing images for region {region_id}"):
        try:
            geotagged_image = GeotaggedImage.load(region_id, file_prefix)
        except Exception:
            print(f"Warning: Failed to load image for: {region_id=}, {file_prefix=}")
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


def find_best_bounding_boxes(
    saliency_map: GeoTIFFData, window_size: int, num_boxes: int
) -> np.ndarray:
    """
    Find the top saliency bounding boxes of the specified size within a saliency map.

    The returned bounding boxes are ordered from highest to lowest saliency, which also corresponds to the class IDs.

    :param saliency_map: The saliency map to generate bounding boxes for.
    :param window_size: The size of the bounding boxes to find in the saliency map. Must be odd.
    :param num_boxes: The number of top saliency boxes to identify.
    :return: A numpy array of shape (num_boxes, 6) containing (centroid_lat, centroid_lon, top_left_lat, top_left_lon,
             bottom_right_lat, bottom_right_lon) for each of the top saliency bounding boxes.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    half_window_size = window_size // 2

    bounding_box_mean_saliencies = uniform_filter(
        saliency_map.image_data[..., 0], size=window_size, mode="constant", cval=0
    )

    # ensure resulting bounding boxes are strictly within the GeoTIFF bounds
    bounding_box_mean_saliencies[:half_window_size, :] = 0
    bounding_box_mean_saliencies[-half_window_size:, :] = 0
    bounding_box_mean_saliencies[:, :half_window_size] = 0
    bounding_box_mean_saliencies[:, -half_window_size:] = 0

    # use argpartition to avoid sorting the entire array
    top_indices = np.argpartition(bounding_box_mean_saliencies, -num_boxes, axis=None)[-num_boxes:]
    centroid_vs, centroid_us = np.unravel_index(top_indices, bounding_box_mean_saliencies.shape)
    # still need to sort the best bounding boxes so that the class IDs are in descending order of saliency
    sort_order = np.argsort(bounding_box_mean_saliencies[centroid_vs, centroid_us])[::-1]
    centroid_vs = centroid_vs[sort_order]
    centroid_us = centroid_us[sort_order]

    top_left_us = centroid_us - half_window_size
    top_left_vs = centroid_vs - half_window_size
    bottom_right_us = centroid_us + half_window_size
    bottom_right_vs = centroid_vs + half_window_size

    inverse_transform = ~saliency_map.transform
    centroid_lon, centroid_lat = inverse_transform * (centroid_us, centroid_vs)
    top_left_lon, top_left_lat = inverse_transform * (top_left_us, top_left_vs)
    bottom_right_lon, bottom_right_lat = inverse_transform * (bottom_right_us, bottom_right_vs)

    bounding_boxes_lat_lon = np.column_stack(
        (centroid_lat, centroid_lon, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon)
    )
    return bounding_boxes_lat_lon


def run_saliency_analysis_for_region(
    region: str, overwrite: bool, gsd: float, bounding_box_size: int, num_boxes: int
) -> None:
    """
    Run the saliency analysis for a single region.

    :param region: The MGRS region to generate the saliency map for.
    :param overwrite: Whether to overwrite the output file if it exists.
    :param gsd: The ground sample distance to use for the saliency map.
    :param bounding_box_size: The side length of the bounding boxes to find in the saliency map, in meters.
    :param num_boxes: The number of top saliency bounding boxes to identify.
    """
    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    region_dir = os.path.join(training_dir, region)
    saliency_map_file = os.path.join(region_dir, SALIENCY_MAP_FILE_NAME)
    bounding_boxes_file = os.path.join(
        training_dir, LandmarkDetector.get_region_bounding_boxes_relative_path(region)
    )
    if os.path.exists(saliency_map_file):
        if not overwrite:
            raise FileExistsError(f"Output file {saliency_map_file} already exists.")
        os.remove(saliency_map_file)
    if os.path.exists(bounding_boxes_file):
        if not overwrite:
            raise FileExistsError(f"Output file {bounding_boxes_file} already exists.")
        os.remove(bounding_boxes_file)

    print(f"Running saliency analysis for region {region}...")

    saliency_map = generate_saliency_map(region_dir, saliency_map_file, gsd, region)
    saliency_map.save()

    window_size = bounding_box_size / gsd
    # round to the nearest odd number of pixels
    window_size = 2 * int(np.rint((window_size - 1) / 2)) + 1
    bounding_boxes_lat_lon = find_best_bounding_boxes(saliency_map, window_size, num_boxes)

    np.savetxt(
        bounding_boxes_file,
        bounding_boxes_lat_lon,
        delimiter=",",
        header="Centroid Latitude,Centroid Longitude,Top-Left Latitude,"
        "Top-Left Longitude,Bottom-Right Latitude,Bottom-Right Longitude",
    )


def main() -> None:
    """
    Script entry point.
    """
    args = parse_args()
    regions = list(set(args.regions) - set(args.skip_regions))
    args.num_processes = min(args.num_processes, len(regions))

    func = partial(
        run_saliency_analysis_for_region,
        overwrite=args.overwrite,
        gsd=args.gsd,
        bounding_box_size=args.bounding_box_size,
        num_boxes=args.num_boxes,
    )
    if args.num_processes > 1:
        with Pool(args.num_processes) as pool:
            pool.map(func, regions)
    else:
        # we don't care about the results, just exhaust the iterator
        list(map(func, regions))


if __name__ == "__main__":
    main()

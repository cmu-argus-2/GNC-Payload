"""
Select bounding boxes from the saliency maps for the specified MGRS regions.

This script expects to find the following contents in the training directory:
- /training_directory
  - /{region}
    - saliency_map.tif

This script will generate/overwrite the following contents in the training directory:
- /training_directory
  - /{region}
    - bounding_boxes.csv
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.ndimage import uniform_filter

from image_simulation.earth_vis import GeoTIFFData
from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.earth_utils import get_mgrs_region_area
from vision_inference.landmark_detector import LandmarkDetector
from VisionTrainingGround.LD.run_saliency_analysis import SALIENCY_MAP_FILE_NAME


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Select bounding boxes for the specified MGRS regions."
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=load_config()["vision"]["salient_mgrs_region_ids"],
        help="MGRS regions to select bounding boxes for.",
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
        help="Number of processes to use for selecting bounding boxes in parallel across the specified regions.",
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


def find_best_bounding_boxes(
    saliency_map: GeoTIFFData, window_size: int, num_boxes: int, decay_exponent: float = 2.0
) -> np.ndarray:
    """
    Find the top saliency bounding boxes of the specified size within a saliency map.

    The returned bounding boxes are ordered from highest to lowest saliency, which also corresponds to the class IDs.

    :param saliency_map: The saliency map to generate bounding boxes for.
    :param window_size: The size of the bounding boxes to find in the saliency map. Must be odd.
    :param num_boxes: The number of top saliency boxes to identify.
    :param decay_exponent: The exponent to use for the decay function. Higher values will reduce the likelihood of
                           selecting nearby bounding boxes.
    :return: A numpy array of shape (num_boxes, 6) containing (centroid_lat, centroid_lon, top_left_lat, top_left_lon,
             bottom_right_lat, bottom_right_lon) for each of the top saliency bounding boxes.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    if np.prod(saliency_map.image_data.shape[:2]) < num_boxes:
        raise ValueError(
            "The number of bounding boxes requested is larger than the number of pixels in the saliency map."
        )
    if window_size > saliency_map.image_data.shape[0] or window_size > saliency_map.image_data.shape[1]:
        raise ValueError(
            "The bounding box size is larger than the saliency map dimensions. "
            "Please select a smaller bounding box size."
        )
    half_window_size = window_size // 2

    bounding_box_mean_saliencies = uniform_filter(
        saliency_map.image_data[..., 0], size=window_size, mode="constant", cval=0
    )

    # ensure resulting bounding boxes are strictly within the GeoTIFF bounds
    bounding_box_mean_saliencies[:half_window_size, :] = 0
    bounding_box_mean_saliencies[-half_window_size:, :] = 0
    bounding_box_mean_saliencies[:, :half_window_size] = 0
    bounding_box_mean_saliencies[:, -half_window_size:] = 0

    delta_us, delta_vs = np.meshgrid(np.arange(window_size), np.arange(window_size))
    distances = np.hypot(delta_us - half_window_size, delta_vs - half_window_size)
    decay_grid = (distances / (np.sqrt(2) * half_window_size)) ** decay_exponent

    centroid_us = np.empty(num_boxes, dtype=int)
    centroid_vs = np.empty(num_boxes, dtype=int)
    for i in range(num_boxes):
        max_u, max_v = np.unravel_index(
            np.argmax(bounding_box_mean_saliencies), bounding_box_mean_saliencies.shape
        )
        if np.isclose(bounding_box_mean_saliencies[max_u, max_v], 0):
            print(
                f"Warning: Fewer than {num_boxes} bounding boxes could be selected. "
                f"Found {i} bounding boxes before stopping."
            )
            centroid_us = centroid_us[:i]
            centroid_vs = centroid_vs[:i]
            break

        centroid_us[i] = max_u
        centroid_vs[i] = max_v

        bounding_box_mean_saliencies[
            max_u - half_window_size : max_u + half_window_size + 1,
            max_v - half_window_size : max_v + half_window_size + 1,
        ] *= decay_grid

    centroid_us = np.array(centroid_us)
    centroid_vs = np.array(centroid_vs)

    top_left_us = centroid_us - half_window_size
    top_left_vs = centroid_vs - half_window_size
    bottom_right_us = centroid_us + half_window_size
    bottom_right_vs = centroid_vs + half_window_size

    inverse_transform = ~saliency_map.transform
    centroid_lat, centroid_lon = inverse_transform * (centroid_us, centroid_vs)
    top_left_lat, top_left_lon = inverse_transform * (top_left_us, top_left_vs)
    bottom_right_lat, bottom_right_lon = inverse_transform * (bottom_right_us, bottom_right_vs)

    bounding_boxes_lat_lon = np.column_stack(
        (centroid_lat, centroid_lon, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon)
    )
    return bounding_boxes_lat_lon


def select_bounding_boxes_for_region(
    region: str, overwrite: bool, bounding_box_size: int, num_boxes: int
) -> None:
    """
    Select bounding boxes for the specified MGRS region.

    :param region: The MGRS region to select bounding boxes for.
    :param overwrite: Whether to overwrite the output file if it exists.
    :param bounding_box_size: The side length of the bounding boxes to find in the saliency map, in meters.
    :param num_boxes: The number of top saliency bounding boxes to identify.
    """
    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    bounding_boxes_file = os.path.join(
        training_dir, LandmarkDetector.get_region_bounding_boxes_relative_path(region)
    )
    if os.path.exists(bounding_boxes_file):
        if not overwrite:
            raise FileExistsError(f"Output file {bounding_boxes_file} already exists.")
        os.remove(bounding_boxes_file)

    saliency_map_file = os.path.join(training_dir, region, SALIENCY_MAP_FILE_NAME)
    if not os.path.exists(saliency_map_file):
        print(f"Saliency map file {saliency_map_file} does not exist. Skipping region {region}.")
        return
    saliency_map = GeoTIFFData.load(saliency_map_file)

    height, width = saliency_map.image_data.shape[:2]
    gsd = np.sqrt(get_mgrs_region_area(region) / (height * width))

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
        select_bounding_boxes_for_region,
        overwrite=args.overwrite,
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

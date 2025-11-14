"""
Landmark analysis.
"""

import argparse
import os
from argparse import Namespace
from multiprocessing import Pool, cpu_count

import numpy as np
from cv2 import cv2
from cv2.typing import MatLike

from utils.earth_utils import get_MGRS_grid

scales = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
CONSOLIDATED_MAPS_PATH = "bm1k_consolidated_maps"
PRIORITIZED_REGIONS_PATH = "prioritized_regions.csv"
LANDMARKS_PATH = "landmarks"
LANDMARKS_VISUALIZATION_PATH = "landmarks/visualizations"
LANDMARKS_PIXEL_PATH = "landmarks/pixels"
CALCULATE_LANDMARKS = None
VISUALIZE_LANDMARKS = None


def parse_args() -> Namespace:
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description="Landmark Analysis")
    parser.add_argument(
        "-r", "--regions", type=str, default=["17R"], help="Region to analyze", nargs="+"
    )
    parser.add_argument(
        "-s",
        "--scales",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        help="Scales to analyze",
    )
    parser.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=0.5,
        help="Fraction of overall max to consider as a landmark",
    )
    parser.add_argument("-c", "--calculate", action="store_true", help="Calculate landmarks")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize landmarks")
    parser.add_argument("--suffix", type=str, default="", help="suffix for the output file")
    return parser.parse_args()


def landmarks_at_scale(region_id: str, scale: int) -> list[list[float]]:
    """
    Docstring.
    """
    im: MatLike = cv2.imread("bm1k_consolidated_maps/" + region_id + ".jpg")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((scale, scale), np.uint8) / (scale * scale)
    # old_sal = None
    overall_max = None
    landmark_list = []
    height, width = im.shape  # pylint: disable=E1101
    while 1:
        maxsum = 0.0
        max_loc = (0, 0)
        for i in range(width - scale + 1):
            for j in range(height - scale + 1):
                cursum = im[j : j + scale, i : i + scale].sum()  # pylint: disable=E1136
                if cursum > maxsum:
                    maxsum = cursum
                    max_loc = (i, j)  # x, y
        # filtered = cv2.filter2D(im, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        # filtered = filtered[scale:-scale, scale:-scale]
        # max_loc = filtered.argmax()
        # max_loc = np.unravel_index(max_loc, filtered.shape)
        # print(max_loc)
        # max_val = filtered.max()
        # pylint: disable=E1137
        im[max_loc[1] : max_loc[1] + scale, max_loc[0] : max_loc[0] + scale] = 0
        # im[max_loc[0]:max_loc[0]+scale, max_loc[1]:max_loc[1]+scale] = 0
        # if old_sal is None:
        #     old_sal = maxsum
        # landmark_list.append([max_loc[0], max_loc[1], scale])
        # else:
        if overall_max is None:
            overall_max = maxsum
        if maxsum <= fraction * overall_max:
            break
        # old_sal = maxsum
        landmark_list.append([max_loc[0], max_loc[1], scale, maxsum])
    return landmark_list


def visualize_landmarks(
    region_id: str, landmark_list: list[tuple[float, float, float, float]]
) -> None:
    """
    Visualize landmarks.
    """
    if not os.path.exists(LANDMARKS_PATH):
        os.makedirs(LANDMARKS_PATH)
    if not os.path.exists(LANDMARKS_VISUALIZATION_PATH):
        os.makedirs(LANDMARKS_VISUALIZATION_PATH)
    im = cv2.imread("bm1k_regions/world_jun/" + region_id + ".jpg")
    for landmark_i in landmark_list:
        x, y, scale, _ = landmark_i
        cv2.rectangle(im, (x, y), (x + scale, y + scale), (0, 255, 0), 1)
    cv2.imwrite(LANDMARKS_VISUALIZATION_PATH + "/" + region_id + "_" + args.suffix + ".jpg", im)
    cv2.imshow(region_id, im)


def get_absolute_landmarks(
    region_id: str, landmark_list: list[tuple[float, float, float, float]]
) -> list[list[float]]:
    """
    Get absolute landmarks.
    """
    bounds = get_MGRS_grid()[region_id]
    reg_im: MatLike = cv2.imread("bm1k_regions/world_jun/" + region_id + ".jpg")
    reg_im_height, reg_im_width = reg_im.shape[:2]  # pylint: disable=E1101
    minx, miny, maxx, maxy = bounds
    lon_per_pix = (maxx - minx) / reg_im_width
    lat_per_pix = (maxy - miny) / reg_im_height
    absolute_landmark_list = []
    for landmark_i in landmark_list:
        x, y, scale, saliency = landmark_i
        x2, y2 = x + scale, y + scale
        x_min_lon = minx + x * lon_per_pix
        x_max_lon = minx + x2 * lon_per_pix
        y_min_lat = maxy - y2 * lat_per_pix
        y_max_lat = maxy - y * lat_per_pix
        x_center_lon = (x_min_lon + x_max_lon) / 2
        y_center_lat = (y_min_lat + y_max_lat) / 2
        absolute_landmark_list.append(
            [
                x_center_lon,
                y_center_lat,
                x_min_lon,
                y_min_lat,
                x_max_lon,
                y_max_lat,
                scale,
                saliency,
            ]
        )
    return absolute_landmark_list


args = parse_args()
regions = args.regions
scales = args.scales
fraction = args.fraction
CALCULATE_LANDMARKS = args.calculate
VISUALIZE_LANDMARKS = args.visualize

if __name__ == "__main__":
    # with open(prioritized_regions_path, 'r') as f:
    #     regions = f.read().split(',')
    # regions = [region for region in regions if region != '']
    if CALCULATE_LANDMARKS:
        pool = Pool(cpu_count())
        for region in regions:
            landmarks = pool.starmap(landmarks_at_scale, [(region, scale) for scale in scales])
            landmarks_array = np.concatenate(landmarks, axis=0)
            np.save(
                LANDMARKS_PIXEL_PATH + "/" + region + "_pixel_landmarks" + args.suffix + ".npy",
                landmarks_array,
            )
            absolute_landmarks = get_absolute_landmarks(
                region,
                np.load(
                    LANDMARKS_PIXEL_PATH + "/" + region + "_pixel_landmarks" + args.suffix + ".npy"
                ),
            )
            np.save(
                LANDMARKS_PATH + "/" + region + "_landmarks" + args.suffix + ".npy",
                absolute_landmarks,
            )
            with open(
                LANDMARKS_PATH + "/" + region + "_landmarks" + args.suffix + ".csv",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    "x_center_lon,y_center_lat,x_min_lon,y_min_lat,x_max_lon,y_max_lat,scale,saliency\n"
                )
                for landmark in absolute_landmarks:
                    f.write(str(landmark)[1:-1] + "\n")
        pool.close()
        pool.join()
    if VISUALIZE_LANDMARKS:
        for region in regions:
            visualize_landmarks(
                region,
                np.load(
                    LANDMARKS_PIXEL_PATH + "/" + region + "_pixel_landmarks" + args.suffix + ".npy"
                ),
            )

"""
Salient Region Analysis script.
"""

import csv

import cv2
import numpy as np
from cv2.typing import MatLike

from utils.earth_utils import get_mgrs_grid

MIN_LAT = -90
MAX_LAT = 90
MIN_LON = -180
MAX_LON = 180
LON_PER_PIX = (MAX_LON - MIN_LON) / 21600
LAT_PER_PIX = (MAX_LAT - MIN_LAT) / 10800


# pylint: disable=too-many-locals
def sareana(
    cloud_im_path: str,
    saliency_im_path: str,
    grid: dict[str, tuple[float, float, float, float]],
    use_sal_only: bool = False,
) -> list[tuple[str, float]]:
    """
    Salient Region analysis function.
    """
    if not use_sal_only:
        cloud_im: MatLike = cv2.imread(cloud_im_path)
        gray_im: MatLike = cv2.cvtColor(cloud_im, cv2.COLOR_BGR2GRAY)
        cloud_im = ~gray_im
        cv2.imwrite("inverse_cloud_map.jpg", cloud_im)
        cloud_im = cloud_im.astype("float32")
        cloud_im = cloud_im / cloud_im.max()

    saliency_im: MatLike = cv2.imread(saliency_im_path)
    # if image is rgb, convert to grayscale
    if len(saliency_im.shape) > 2 and saliency_im.shape[2] == 3:
        saliency_im = cv2.cvtColor(saliency_im, cv2.COLOR_BGR2GRAY)
    saliency_im = saliency_im.astype("float32")
    saliency_im = saliency_im / saliency_im.max()

    im_height, im_width = saliency_im.shape[:2]

    if not use_sal_only:
        cloud_im_resized = cv2.resize(cloud_im, (im_width, im_height))
        sareana_im = np.zeros((im_height, im_width, 3), dtype=np.uint8)
        sareana_im[:, :, 0] = (255 * cloud_im_resized.copy()).astype("uint8")
        sareana_im[:, :, 1] = (255 * saliency_im.copy()).astype("uint8")
        cv2.imwrite("sareana.jpg", sareana_im)
        sareana_mul = cloud_im_resized * saliency_im
    else:
        sareana_im = np.zeros((im_height, im_width, 3), dtype=np.uint8)
        sareana_im[:, :, 1] = (255 * saliency_im.copy()).astype("uint8")
        cv2.imwrite("saliency_only.jpg", sareana_im)
        sareana_mul = saliency_im

    sareana_mul = sareana_mul / sareana_mul.max()
    sareana_mul_im = (sareana_mul.copy() * 255).astype("uint8")
    sareana_im[:, :, 2] = sareana_mul_im
    cv2.imwrite("sareana_mul.jpg", sareana_im)
    cv2.imwrite("sareana_only_mul.jpg", sareana_mul_im)

    reg_sareana = sareana_mul.copy()
    reg_sals = {}

    for key, value in grid.items():
        left, bottom, right, top = value
        left = left - MIN_LON
        right = right - MIN_LON
        bottom = 180 - (bottom - MIN_LAT)
        top = 180 - (top - MIN_LAT)
        left_px = int(left / LON_PER_PIX)
        right_px = int(right / LON_PER_PIX)
        top_px = int(bottom / LAT_PER_PIX)
        bottom_px = int(top / LAT_PER_PIX)
        region_im = sareana_mul[bottom_px:top_px, left_px:right_px]

        if key[-1] == "X" or key[-1] == "W" or key[-1] == "C" or key[-1] == "D":
            region_sal = 0.0
        else:
            region_sal = region_im.sum() / (region_im.shape[0] * region_im.shape[1])

        reg_sals[key] = region_sal
        reg_sareana[bottom_px:top_px, left_px:right_px] = region_sal

    reg_sareana = reg_sareana / reg_sareana.max() * 255
    reg_sareana = reg_sareana.astype("uint8")
    cv2.imwrite("reg_sareana.jpg", reg_sareana)

    reg_sals_sorted = sorted(reg_sals.items(), key=lambda x: x[1], reverse=True)

    with open("prioritized_regions.csv", "w", encoding="utf-8") as f2:
        for key2, val in reg_sals_sorted:
            if val == 0.0:
                break
            f2.write(key2 + ",")

    return reg_sals_sorted


if __name__ == "__main__":
    mgrs_grid = get_mgrs_grid()
    USE_SALIENCY_ONLY = True

    # Set use_saliency_only to True if you want to use only saliency map data
    sorted_reg_sals = sareana(
        "world_saliency.jpg", "world_saliency.jpg", mgrs_grid, USE_SALIENCY_ONLY
    )

    print(sorted_reg_sals)
    NP_FILE_NAME = (
        "sorted_region_saliencys_no_cloud.npy"
        if USE_SALIENCY_ONLY
        else "sorted_region_saliencys.npy"
    )
    np.save(NP_FILE_NAME, sorted_reg_sals)

    with open(NP_FILE_NAME[:-4] + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Value"])  # Header
        writer.writerows(sorted_reg_sals)  # Write data

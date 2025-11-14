"""
Script to make grid images
"""

import os
from multiprocessing import Pool

import cv2
import numpy as np
from cv2.typing import MatLike

from utils.earth_utils import get_MGRS_grid

FOLDER = "bm1k"
OUTFOLDER = "bm1k_regions"
grid = get_MGRS_grid()
MIN_LAT = -90
MAX_LAT = 90
MIN_LON = -180
MAX_LON = 180
LON_PER_PIX = (MAX_LON - MIN_LON) / 21600
LAT_PER_PIX = (MAX_LAT - MIN_LAT) / 10800
REGIONS_FOLDER = "bm1k_consolidated_maps"

SAVE_REGIONS = True
RECOMBINE = False


def save_regions(file: str) -> None:
    """
    Save regions.
    """
    im: MatLike = cv2.imread(os.path.join(FOLDER, file))
    for key, value in grid.items():
        left, bottom, right, top = value
        left = left - MIN_LON
        right = right - MIN_LON
        bottom = 180 - (bottom - MIN_LAT)
        top = 180 - (top - MIN_LAT)
        left_px = left / LON_PER_PIX
        right_px = right / LON_PER_PIX
        top_px = bottom / LAT_PER_PIX
        bottom_px = top / LAT_PER_PIX
        print(bottom_px, top_px, left_px, right_px)
        # pylint: disable=E1136
        reg_im = im[int(bottom_px) : int(top_px), int(left_px) : int(right_px)]
        cv2.imwrite(os.path.join(OUTFOLDER, file[:-4], key + ".jpg"), reg_im)
        print(file, key, "done")
    print(file, "done")


def recombine_regions() -> None:
    """
    Recombine regions.
    """
    out_im = np.zeros((10800, 21600), dtype=np.uint8)
    for file in os.listdir(REGIONS_FOLDER):
        im = cv2.imread(os.path.join(REGIONS_FOLDER, file))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        reg = file[:-4]
        left, bottom, right, top = grid[reg]
        left = left - MIN_LON
        right = right - MIN_LON
        bottom = 180 - (bottom - MIN_LAT)
        top = 180 - (top - MIN_LAT)
        left_px = left / LON_PER_PIX
        right_px = right / LON_PER_PIX
        top_px = bottom / LAT_PER_PIX
        bottom_px = top / LAT_PER_PIX
        out_im[int(bottom_px) : int(top_px), int(left_px) : int(right_px)] = im
        print(file, "done")
    cv2.imwrite("world_saliency.jpg", out_im)


if __name__ == "__main__":
    if SAVE_REGIONS:
        if not os.path.exists(OUTFOLDER):
            os.mkdir(OUTFOLDER)
        months = os.listdir(FOLDER)
        for month in months:
            if not os.path.exists(os.path.join(OUTFOLDER, month[:-4])):
                os.mkdir(os.path.join(OUTFOLDER, month[:-4]))
        with Pool(3) as p:
            p.map(save_regions, months)
    if RECOMBINE:
        recombine_regions()

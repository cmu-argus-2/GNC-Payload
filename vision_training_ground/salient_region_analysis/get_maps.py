"""
Get region and clpud maps.
"""

import os
from multiprocessing import Pool

import cv2
import numpy as np
from cv2.typing import MatLike
from utils.earth_utils import get_mgrs_grid

FOLDER = "bm1k_regions"
OUTFOLDER = "bm1k_maps"
CONSFOLDER = "bm1k_consolidated_maps"
CLOUDFOLDER = "cloud_maps"
# pylint: disable=I1101
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
SAVEMAP = False
CONSOLIDATE = True
CLOUDS = False
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def save_map(file: str, subfolder: str) -> cv2.typing.MatLike:
    """
    Save saliency map.
    """
    im = cv2.imread(FOLDER + "/" + subfolder + "/" + file)
    _, saliency_map = saliency.computeSaliency(im)
    saliency_map_int = (saliency_map * 255).astype("uint8")
    cv2.imwrite(os.path.join(OUTFOLDER, subfolder, file), saliency_map_int)
    return saliency_map


def consolidate_region_maps(region: str) -> None:
    """
    Consolidate region maps.
    """
    reg_im = None
    for month in MONTHS:
        if reg_im is None:
            im: MatLike = cv2.imread(os.path.join(OUTFOLDER, "world_" + month, region + ".jpg"))
            reg_im = np.zeros(im.shape, dtype=np.float32)  # pylint: disable=E1101
        else:
            reg_im += cv2.imread(os.path.join(OUTFOLDER, "world_" + month, region + ".jpg"))
    reg_im = reg_im / len(MONTHS)
    reg_im = reg_im.astype("uint8")
    cv2.imwrite(os.path.join(CONSFOLDER, region + ".jpg"), reg_im)
    print(region, "done")


def consolidate_clouds() -> None:
    """
    Consolidate clouds.
    """
    cloudmaps = os.listdir(CLOUDFOLDER)
    conscloud = None
    for cloudmap in cloudmaps:
        if conscloud is None:
            im = cv2.imread(os.path.join(CLOUDFOLDER, cloudmap))
            conscloud = np.zeros(im.shape, dtype=np.float32)  # pylint: disable=E1101
        else:
            conscloud += cv2.imread(os.path.join(CLOUDFOLDER, cloudmap))
    conscloud = conscloud / len(cloudmaps)
    conscloud = conscloud.astype("uint8")
    cv2.imwrite("consolidated_cloudmap.jpg", conscloud)


def one_month(month: str) -> None:
    """
    Creates folder for and calls the function to save the map for the input month.
    """
    if not os.path.exists(os.path.join(OUTFOLDER, "world_" + month)):
        os.mkdir(os.path.join(OUTFOLDER, "world_" + month))
    for file in os.listdir(os.path.join(FOLDER, "world_" + month)):
        save_map(file, "world_" + month)
    print(month, "done")


if __name__ == "__main__":
    grid = get_mgrs_grid()
    if not os.path.exists(OUTFOLDER):
        os.mkdir(OUTFOLDER)
    if SAVEMAP:
        with Pool(12) as p:
            p.map(one_month, MONTHS)
    if CONSOLIDATE:
        if not os.path.exists(CONSFOLDER):
            os.mkdir(CONSFOLDER)
        for key, value in grid.items():
            consolidate_region_maps(key)
            print(key, "done")
    if CLOUDS:
        consolidate_clouds()
        print("cloudmap done")

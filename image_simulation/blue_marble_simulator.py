"""
Code for simulating the colors of the Earth's surface as seen from space using the Blue Marble Next Generation dataset.

The dataset is divided into directories by month, and each month directory contains an equirectangular projection
of the Earth's surface divided into 8 tiles arranged in a 2x4 grid as follows:
A1 B1 C1 D1
A2 B2 C2 D2
"""

import os
import cv2
from functools import lru_cache

import numpy as np

from utils.config_utils import USER_CONFIG_PATH, load_config


MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
IMG_LAT_BOUNDS = {
    "1": (0, 90),
    "2": (-90, 0),
}
IMG_LON_BOUNDS = {
    "A": (-180, -90),
    "B": (-90, 0),
    "C": (0, 90),
    "D": (90, 180),
}


@lru_cache(maxsize=1)
def get_blue_marble_img(month: str, img_name: str) -> np.ndarray:
    path = os.path.join(load_config(USER_CONFIG_PATH)["blue_marble_directory"], month, f"{img_name}.png")
    assert os.path.exists(path), f"Blue Marble image not found: {path}"

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def query_blue_marble_pixel_colors(lat_lon: np.ndarray, month: str | None = None) -> np.ndarray:
    """
    Query the colors of the pixels at the given latitudes and longitudes.

    :param lat_lon: A numpy array of shape (..., 2) containing the latitudes and longitudes of the pixels to query.
    :param month: The month to simulate. If None, a random month will be chosen.
    :return: A numpy array of shape (..., 3) containing the RGB values of the pixels.
    """
    assert lat_lon.shape[-1] == 2, "The last dimension of lat_lon must be 2."
    if lat_lon.ndim == 1:
        # special case for single query
        return query_blue_marble_pixel_colors(lat_lon[np.newaxis, :], month)[0, :]

    if month is None:
        month = np.random.choice(MONTH_NAMES)

    shape_prefix = lat_lon.shape[:-1]
    lat_lon = lat_lon.reshape(-1, 2)

    img_letters = np.full(lat_lon.shape[0], "", dtype=str)
    for letter in "ABCD":
        min_lon, max_lon = IMG_LON_BOUNDS[letter]
        img_letters[(min_lon <= lat_lon[:, 1]) & (lat_lon[:, 1] < max_lon)] = letter
    assert np.all(img_letters != ""), "Longitude out of bounds."

    img_numbers = np.where(lat_lon[:, 0] >= 0, "1", "2")
    img_names = np.char.add(img_letters, img_numbers)

    pixel_colors = np.zeros((lat_lon.shape[0], 3), dtype=np.uint8)
    for img_name in set(img_names):
        img = get_blue_marble_img(month, img_name)
        height, width = img.shape[:2]

        min_lat, max_lat = IMG_LAT_BOUNDS[img_name[1]]
        min_lon, max_lon = IMG_LON_BOUNDS[img_name[0]]

        img_lat_lon = lat_lon[img_names == img_name, :]
        us = (img_lat_lon[:, 1] - min_lon) / (max_lon - min_lon) * width
        vs = (img_lat_lon[:, 0] - min_lat) / (max_lat - min_lat) * height
        us = np.floor(us).astype(int)
        vs = np.floor(vs).astype(int)
        us[us == width] = width - 1
        vs[vs == height] = height - 1

        pixel_colors[img_names == img_name, :] = img[vs, us, :]

    return pixel_colors.reshape(shape_prefix + (3,))

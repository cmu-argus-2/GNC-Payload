"""
Module to simulate and visualize Earth images from satellite data.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import ClassVar, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
import rasterio
from affine import Affine
from rasterio.crs import CRS

from sensors.camera_model import CameraModel
from utils.config_utils import USER_CONFIG_PATH, load_config

# pylint: disable=import-error
from utils.earth_utils import (
    calculate_mgrs_zones,
    ecef_to_lat_lon,
    get_MGRS_grid,
    intersect_ellipsoid,
)
from vision_inference.frame import Frame
from image_simulation.blue_marble_simulator import query_blue_marble_pixel_colors


@dataclass
class GeoTIFFData:
    """
    Dataclass to store the data contained in a GeoTIFF file.

    Attributes:
        image_path: The path to the GeoTIFF file.
        image_data: The image data contained in the GeoTIFF file.
        transform: The affine transformation matrix for the GeoTIFF file which maps a tuple of (latitudes, longitudes)
                   to a tuple of (us, vs) (i.e. pixel coordinates).
    """

    SWAP_INPUTS_AFFINE: ClassVar[Affine] = Affine(a=0, b=1, c=0, d=1, e=0, f=0)
    SUPPORTED_DTYPES: ClassVar[Tuple[type, ...]] = (np.uint8, np.float32)
    EPSG_4326_CRS: ClassVar[CRS] = CRS.from_epsg(4326)

    image_path: str
    image_data: np.ndarray
    transform: Affine

    @staticmethod
    def load(file_path: str) -> "GeoTIFFData":
        """
        Load GeoTIFFData from a file.

        Parameters:
            file_path: Path to the GeoTIFF file.

        Returns:
            GeoTIFFData: The GeoTIFFData contained in the file.
        Raises:
            ValueError: If the GeoTIFF file contains a coordinate reference system other than GeoTIFFData.EPSG_4326_CRS,
                        or if the data type of the image is not in GeoTIFFData.SUPPORTED_DTYPES.
        """
        with rasterio.open(file_path) as src:
            if src.crs != GeoTIFFData.EPSG_4326_CRS:
                raise ValueError(
                    f"GeoTIFF file located at {file_path} contains "
                    f"an unsupported coordinate reference system: {src.crs}"
                )
            image_data = src.read()
            transform: Affine = src.transform

        if image_data.dtype not in GeoTIFFData.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported data type {image_data.dtype}. Supported data types are: "
                f"{', '.join(str(dtype) for dtype in GeoTIFFData.SUPPORTED_DTYPES)}."
            )

        # convert from (channels, height, width) to (height, width, channels)
        image_data = np.moveaxis(image_data, 0, -1)

        # switch from (u, v) -> (lon, lat) to (lon, lat) -> (u, v)
        transform = ~transform
        # switch from (lon, lat) -> (u, v) to (lat, lon) -> (u, v)
        transform = transform * GeoTIFFData.SWAP_INPUTS_AFFINE

        return GeoTIFFData(file_path, image_data, transform)

    def save(self) -> None:
        """
        Save the contents of this GeoTIFFData object to the underlying file specified by self.image_path.
        Note that this will overwrite any existing file at that location.

        Note that this assumes that self.transform maps to pixel coordinates from the EPSG:4326 coordinate reference
        system, which corresponds to (latitude, longitude) coordinates in degrees using the WGS 84 ellipsoid.
        """
        assert self.dtype in GeoTIFFData.SUPPORTED_DTYPES, (
            f"Unsupported data type {self.dtype}. Supported data types are: "
            f"{', '.join(str(dtype) for dtype in GeoTIFFData.SUPPORTED_DTYPES)}."
        )
        height, width, num_channels = self.image_data.shape

        # convert from (height, width, channels) to (channels, height, width)
        image_data = np.moveaxis(self.image_data, -1, 0)

        # switch from (lat, lon) -> (u, v) to (lon, lat) -> (u, v)
        transform = self.transform * GeoTIFFData.SWAP_INPUTS_AFFINE
        # switch from (lon, lat) -> (u, v) to (u, v) -> (lon, lat)
        transform = ~transform

        metadata = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": num_channels,
            "dtype": self.dtype,
            "crs": GeoTIFFData.EPSG_4326_CRS,
            "transform": transform,
        }
        with rasterio.open(self.image_path, "w", **metadata) as dst:
            dst.write(image_data)

    @property
    def num_channels(self) -> int:
        """
        Get the number of channels in the GeoTIFF data.

        Returns:
            The number of channels in the GeoTIFF data.
        """
        return self.image_data.shape[-1]

    @property
    def dtype(self) -> np.dtype:
        """
        Get the data type of the GeoTIFF data.

        Returns:
            The data type of the GeoTIFF data.
        """
        return self.image_data.dtype

    def remap_to_mgrs_region(self, region_id: str) -> None:
        """
        Remap this GeoTIFFData to represent the specified MGRS region.
        Note that this overwrites the current transform, which is loaded from the underlying GeoTIFF file by default.

        :param region_id: The MGRS region ID to remap this GeoTIFFData to.
        """
        height, width, _ = self.image_data.shape
        min_lon, min_lat, max_lon, max_lat = get_MGRS_grid()[region_id]
        scale_u = width / (max_lon - min_lon)
        scale_v = height / (max_lat - min_lat)

        # maps (lat, lon) to (u, v) (i.e. width, height)
        self.transform = Affine(
            # u = a * lat + b * lon + c, lon = min_lon -> u = 0, lon = max_lon -> u = width
            a=0,
            b=scale_u,
            c=-min_lon * scale_u,
            # v = d * lat + e * lon + f, lat = min_lat -> v = height, lat = max_lat -> v = 0
            d=-scale_v,
            e=0,
            f=max_lat * scale_v,
        )

    def get_pixel_coordinates(
        self, lat_lon: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the pixel coordinates corresponding to the given latitudes and longitudes.
        A mask is also returned to indicate which output pixel coordinates contain data that is in bounds.

        :param lat_lon: A numpy array of shape (..., 2) containing the latitudes and longitudes to query.
        :return: A Tuple containing:
                 - A numpy array of shape lat_lon.shape[:-1] containing the horizontal pixel coordinates, u.
                 - A numpy array of shape lat_lon.shape[:-1] containing the vertical pixel coordinates, v.
                 - A numpy array of shape lat_lon.shape[:-1] indicating which pixel coordinates contain data that is in
                   bounds.
        """
        assert lat_lon.shape[-1] == 2, "lat_lon must have shape (..., 2)."

        shape_prefix = lat_lon.shape[:-1]
        lat_lon = lat_lon.reshape(-1, 2)

        us, vs = self.transform * tuple(lat_lon.T)

        height, width, _ = self.image_data.shape
        us = np.rint(us).astype(int).reshape(shape_prefix)
        vs = np.rint(vs).astype(int).reshape(shape_prefix)
        us[us == width] = width - 1
        vs[vs == height] = height - 1

        valid_mask = (vs >= 0) & (vs < height) & (us >= 0) & (us < width)
        return us, vs, valid_mask

    def query_pixel_colors(self, lat_lon: np.ndarray) -> np.ndarray:
        """
        Query pixel colors from this GeoTIFFData for a set of latitudes and longitudes.

        The pixel colors' channels will be returned in the same order as the GeoTIFF data, which should be in the order
        (red, green, blue).

        :param lat_lon: A numpy array of shape (..., 2) containing the latitudes and longitudes to query.
        :return: A numpy array of shape lat_lon.shape[:-1] + (self.num_channels,) containing the pixel values.
        """
        us, vs, valid_mask = self.get_pixel_coordinates(lat_lon)

        image = np.zeros(lat_lon.shape[:-1] + (self.num_channels,), dtype=self.image_data.dtype)
        image[valid_mask, :] = self.image_data[vs[valid_mask], us[valid_mask], :]
        return image


class GeoTIFFCache:
    """
    This class is responsible for loading and caching GeoTIFF data for Earth image simulation.

    Attributes:
        FALLBACK_GEOTIFF_FOLDER: Default folder containing GeoTIFF files. Only used if the user configuration file is not found.
    """

    FALLBACK_GEOTIFF_FOLDER = "/home/argus/eedl_images/"

    def __init__(self, geotiff_folder: str | None = None, max_cache_size: int | None = 58):
        """
        Initialize the GeoTIFF cache.

        Parameters:
            geotiff_folder: Path to the folder containing GeoTIFF files.
            max_cache_size: Maximum number of regions to maintain in the cache.
                            Set to 0 to disable caching. Set to None for unlimited size.
                            The default value was chosen via compute_max_visible_regions in test_earth_vis.py.
        """
        self.geotiff_folder = (
            geotiff_folder
            if geotiff_folder is not None
            else GeoTIFFCache.get_default_geotiff_folder()
        )
        GeoTIFFCache.validate_salient_region_data_exists(self.geotiff_folder)

        # Dynamically wrap the member function with an LRU cache
        # This also ensures that each instance has its own cache and prevents the need to call hash(self) inside the
        # cache implementation
        self.load_geotiff_data = lru_cache(maxsize=max_cache_size)(self.load_geotiff_data)

    @staticmethod
    def get_default_geotiff_folder() -> str:
        """
        Get the default GeoTIFF folder from the user configuration file.

        Returns:
            The default GeoTIFF folder.
        """
        if os.path.exists(USER_CONFIG_PATH):
            return load_config(USER_CONFIG_PATH)["geotiff_folder"]

        print("User configuration file not found. Using fallback GeoTIFF folder.")
        return EarthImageSimulator.FALLBACK_GEOTIFF_FOLDER

    def simulate_image(
        self, position_ecef: np.ndarray, ecef_R_body: np.ndarray, camera_model: CameraModel
    ) -> Frame:
        """
        Simulate an Earth image given the satellite position, attitude, and camera model.

        Parameters:
            position_ecef: A numpy array of shape (3,) representing the satellite position in ECEF coordinates.
            ecef_R_body: A numpy array of shape (3, 3) representing the rotation matrix from body to ECEF coordinates.
            camera_model: The camera model to use to simulate the image.

        Returns:
            The simulated Frame object.
        """
        frame, _ = self.simulate_image_for_training(position_ecef, ecef_R_body, camera_model)
        return frame

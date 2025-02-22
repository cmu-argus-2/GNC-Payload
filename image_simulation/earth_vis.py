"""
Module to simulate and visualize Earth images from satellite data.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from affine import Affine

from sensors.camera_model import CameraModel
from utils.config_utils import USER_CONFIG_PATH, load_config

# pylint: disable=import-error
from utils.earth_utils import calculate_mgrs_zones, ecef_to_lat_lon
from vision_inference.frame import Frame


@dataclass
class GeoTIFFData:
    """
    Dataclass to store the data contained in a GeoTIFF file.

    Attributes:
        image_path: The path to the GeoTIFF file.
        image_data: The image data contained in the GeoTIFF file.
        transform: The affine transformation matrix for the GeoTIFF file which maps a tuple of (longitudes, latitudes)
                   to a tuple of (us, vs) (i.e. pixel coordinates).
    """

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
        """
        with rasterio.open(file_path) as src:
            image_data = src.read()
            transform = src.transform

        # convert from (channels, height, width) to (height, width, channels)
        image_data = np.moveaxis(image_data, 0, -1)
        inverse_transform = ~transform
        return GeoTIFFData(file_path, image_data, inverse_transform)

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

    def query_pixel_colors(self, lat_lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query pixel colors from this GeoTIFFData for a set of latitudes and longitudes.

        :param lat_lon: A numpy array of shape (..., 2) containing the latitudes and longitudes to query.
        :return: A Tuple containing:
                 - A numpy array of shape lat_lon.shape[:-1] + (self.num_channels,) containing the pixel values.
                 - A numpy array of shape lat_lon.shape[:-1] indicating which output pixels contain valid data.
        """
        assert lat_lon.shape[-1] == 2, "lat_lon must have shape (..., 2)."

        shape_prefix = lat_lon.shape[:-1]
        lat_flat, lon_flat = lat_lon.reshape(-1, 2).T

        us, vs = self.transform * (lon_flat, lat_flat)
        us = np.floor(us).astype(int)
        vs = np.floor(vs).astype(int)

        height, width, num_channels = self.image_data.shape
        valid_mask = (vs >= 0) & (vs < height) & (us >= 0) & (us < width)

        num_pixels = np.prod(shape_prefix)
        image_flat = np.zeros((num_pixels, num_channels), dtype=self.image_data.dtype)
        image_flat[valid_mask, :] = self.image_data[vs[valid_mask], us[valid_mask], :]

        return image_flat.reshape(*shape_prefix, num_channels), valid_mask.reshape(shape_prefix)


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
        GeoTIFFCache.validate_region_folders_exist(self.geotiff_folder)

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
        return GeoTIFFCache.FALLBACK_GEOTIFF_FOLDER

    @staticmethod
    def validate_region_folders_exist(geotiff_folder: str) -> None:
        """
        Check if all salient region folders exist in the specified GeoTIFF folder.

        Parameters:
            geotiff_folder: Path to the folder containing GeoTIFF files.

        Raises:
            FileNotFoundError: If one or more region folders are not found.
        """
        salient_region_ids = load_config()["vision"]["salient_mgrs_region_ids"]

        all_region_folders_exist = True
        for region in salient_region_ids:
            region_folder = os.path.join(geotiff_folder, region)
            if not os.path.exists(region_folder):
                print(f"WARNING: Region folder '{region_folder}' not found.")
                all_region_folders_exist = False
        if all_region_folders_exist:
            print("All salient region folders found!")
        else:
            raise FileNotFoundError("One or more region folders not found.")

    def load_geotiff_data(self, region: str) -> GeoTIFFData | None:
        """
        Load GeoTIFF data for a specific region.

        Note that this function is dynamically wrapped with an LRU cache in the constructor, so it will cache its
        outputs for recent regions. This makes it likely that temporally adjacent images will be loaded from the cache,
        resulting in consistent image appearance.

        :param region: The MGRS region to load data for.
        :return: A GeoTIFFData object, or None if there is no data for the specified region.
        """
        region_folder = os.path.join(self.geotiff_folder, region)
        if not os.path.exists(region_folder):
            return None
        region_files = os.listdir(region_folder)
        if not region_files:
            return None

        selected_file = np.random.choice(region_files)
        file_path = os.path.join(region_folder, selected_file)
        return GeoTIFFData.load(file_path)

    def clear_cache(self) -> None:
        """
        Clear the GeoTIFF cache.
        """
        self.load_geotiff_data.cache_clear()


class EarthImageSimulator:
    """
    Simulator for simulating Earth images from downloaded GeoTIFF files, accounting for satellite position and orientation.
    """

    def __init__(self, geotiff_cache: GeoTIFFCache | None = None):
        """
        Initialize the Earth image simulator.

        Parameters:
            geotiff_cache: The GeoTIFFCache to use. If None, a default GeoTIFFCache will be created.
        """
        self.cache = geotiff_cache if geotiff_cache is not None else GeoTIFFCache()

    def simulate_image_for_training(
        self, position_ecef: np.ndarray, ecef_R_body: np.ndarray, camera_model: CameraModel
    ) -> Tuple[Frame, np.ndarray, np.ndarray]:
        """
        Simulate an Earth image given the satellite position, attitude, and camera model.
        This method also returns the MGRS regions and latitudes/longitudes for each pixel.

        Parameters:
            position_ecef: A numpy array of shape (3,) representing the satellite position in ECEF coordinates.
            ecef_R_body: A numpy array of shape (3, 3) representing the rotation matrix from body to ECEF coordinates.
            camera_model: The camera model to use to simulate the image.

        Returns:
            A Tuple containing:
            - The simulated Frame object.
            - A numpy array of shape CameraModel.RESOLUTION containing the MGRS regions for each pixel,
              or None if the pixel does not correspond to any MGRS region.
            - A numpy array of shape CameraModel.RESOLUTION + (2,) containing the latitudes and longitudes for each
              pixel, or np.nan if the pixel does not correspond to any MGRS region.
        """
        ray_directions_body = camera_model.ray_directions_body()
        ray_directions_ecef = ray_directions_body @ ecef_R_body.T

        camera_position_ecef = camera_model.get_camera_position(position_ecef, ecef_R_body)
        intersection_points = intersect_ellipsoid(ray_directions_ecef, camera_position_ecef)
        lat_lon = ecef_to_lat_lon(intersection_points)

        # TODO: see if we can avoid calculating this for every pixel
        mgrs_regions = calculate_mgrs_zones(lat_lon)
        present_regions = np.unique(mgrs_regions[mgrs_regions != None])

        image = np.zeros(CameraModel.OUTPUT_SHAPE, dtype=CameraModel.DTYPE)
        valid_mask = np.zeros(CameraModel.RESOLUTION, dtype=bool)
        for region in present_regions:
            geotiff_data = self.cache.load_geotiff_data(region)
            if geotiff_data is None:
                continue

            assert geotiff_data.num_channels == CameraModel.NUM_CHANNELS, (
                f"The GeoTIFF data located at '{geotiff_data.image_path}' does not have {CameraModel.NUM_CHANNELS} "
                f"channels as expected in the camera model."
            )
            assert geotiff_data.dtype == CameraModel.DTYPE, (
                f"The GeoTIFF data located at {geotiff_data.image_path} does not have a dtype of "
                f"{CameraModel.DTYPE} as expected in the camera model."
            )

            region_mask = (mgrs_regions == region).reshape(CameraModel.RESOLUTION)
            region_image, region_valid_mask = geotiff_data.query_pixel_colors(lat_lon[region_mask])

            image[region_mask] = region_image
            valid_mask[region_mask] |= region_valid_mask

        # TODO: Use ocean imagery for pixels that do not belong to any MGRS region

        return (
            Frame(image, camera_model.camera_name, datetime.now()),
            mgrs_regions,
            lat_lon,
        )

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
        frame, *_ = self.simulate_image_for_training(position_ecef, ecef_R_body, camera_model)
        return frame

    def display_image(self, image):
        """
        Display the simulated image.

        Parameters:
            image (np.ndarray): Simulated RGB image.
        """
        plt.imshow(image)
        plt.axis("off")
        plt.show()


def intersect_ellipsoid(ray_directions, satellite_position, a=6378137.0, b=6356752.314245):
    """
    Vectorized computation of ray intersections with the WGS84 ellipsoid.

    Parameters:
        ray_directions (np.ndarray): Array of ray directions (Nx3).
        satellite_position (np.ndarray): Satellite position in ECEF (3,).
        a (float): Semi-major axis of the WGS84 ellipsoid (meters).
        b (float): Semi-minor axis of the WGS84 ellipsoid (meters).

    Returns:
        np.ndarray: Intersection points (Nx3), or NaN for rays that miss.
    """
    H, W, _ = ray_directions.shape
    ray_directions_flat = ray_directions.reshape(-1, 3)

    A = (
        ray_directions_flat[:, 0] ** 2 / a**2
        + ray_directions_flat[:, 1] ** 2 / a**2
        + ray_directions_flat[:, 2] ** 2 / b**2
    )
    B = 2 * (
        satellite_position[0] * ray_directions_flat[:, 0] / a**2
        + satellite_position[1] * ray_directions_flat[:, 1] / a**2
        + satellite_position[2] * ray_directions_flat[:, 2] / b**2
    )
    C = (
        satellite_position[0] ** 2 / a**2
        + satellite_position[1] ** 2 / a**2
        + satellite_position[2] ** 2 / b**2
        - 1
    )
    discriminant = B**2 - 4 * A * C

    # Initialize intersection points as NaN
    intersection_points_flat = np.full_like(ray_directions_flat, np.nan)

    valid_mask = discriminant >= 0
    if np.any(valid_mask):
        # Compute roots of the quadratic equation
        sqrt_discriminant = np.sqrt(discriminant[valid_mask])
        t1 = (-B[valid_mask] - sqrt_discriminant) / (2 * A[valid_mask])
        t2 = (-B[valid_mask] + sqrt_discriminant) / (2 * A[valid_mask])

        # Choose the smallest positive t
        t = np.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
        t = np.where(t > 0, t, np.nan)  # Filter out negative t values

        # Calculate intersection points
        valid_ray_directions = ray_directions_flat[valid_mask]
        intersection_points_flat[valid_mask] = (
            t[:, None] * valid_ray_directions + satellite_position
        )
    # Reshape intersection points back to original ray grid shape
    intersection_points = intersection_points_flat.reshape(H, W, 3)
    return intersection_points

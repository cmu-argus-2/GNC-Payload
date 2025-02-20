"""
Module to simulate and visualize Earth images from satellite data.
"""

import os
from datetime import datetime
from typing import Tuple
from functools import lru_cache
from dataclasses import dataclass

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
        image_data: The image data contained in the GeoTIFF file.
        transform: The affine transformation matrix for the GeoTIFF file.
    """
    image_data: np.ndarray
    transform: Affine


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
        self.geotiff_folder = geotiff_folder if geotiff_folder is not None else GeoTIFFCache.get_default_geotiff_folder()
        GeoTIFFCache.validate_region_folders_exist(self.geotiff_folder)

        # Dynamically wrap the member function with an LRU cache
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
        with rasterio.open(file_path) as src:
            image_data = src.read()
            image_data = np.moveaxis(image_data, 0, -1)
            transform = src.transform
        return GeoTIFFData(image_data, transform)

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
            - A numpy array containing the MGRS regions for each pixel,
              or None if the pixel does not correspond to any MGRS region.
            - A numpy array containing the latitudes and longitudes for each pixel,
              or np.nan if the pixel does not correspond to any MGRS region.
        """
        # Generate ray directions in ECEF frame
        ray_directions_body = camera_model.ray_directions()
        ray_directions_ecef = ray_directions_body @ ecef_R_body.T

        # Intersect rays with the Earth
        camera_position_ecef = camera_model.get_camera_position(position_ecef, ecef_R_body)
        intersection_points = intersect_ellipsoid(ray_directions_ecef, camera_position_ecef)

        # Convert intersection points to lat/lon
        lat_lon = ecef_to_lat_lon(intersection_points)

        # Flatten latitude/longitude grid
        lat_lon_flat = lat_lon.reshape(-1, 2)
        latitudes = lat_lon_flat[:, 0]
        longitudes = lat_lon_flat[:, 1]

        # Calculate present MGRS regions
        mgrs_regions = calculate_mgrs_zones(latitudes, longitudes)
        present_regions = np.unique([region for region in mgrs_regions if region is not None])

        # Initialize full image with zeros
        width, height = CameraModel.RESOLUTION
        pixel_colors_full = np.zeros((height, width, 3), dtype=np.uint8)

        # Load and assign data for each region
        for region in present_regions:
            geotiff_data = self.cache.load_geotiff_data(region)
            if geotiff_data is None:
                continue

            # Mask for the current region
            region_mask = (mgrs_regions == region).reshape(height, width)

            # Skip if no pixels belong to this region
            if not np.any(region_mask):
                continue

            # Query pixel colors for the region
            pixel_colors_region = query_pixel_colors(
                latitudes[region_mask.flatten()], longitudes[region_mask.flatten()], geotiff_data.data, geotiff_data.trans
            )

            # Assign pixel values to the full image
            pixel_colors_full[region_mask] = pixel_colors_region

        return Frame(pixel_colors_full, camera_model.camera_name, datetime.now()), mgrs_regions, lat_lon

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


def query_pixel_colors(latitudes, longitudes, image_data, trans):
    latitudes_flat = latitudes.flatten()
    longitudes_flat = longitudes.flatten()

    inverse_transform = ~trans

    cols, rows = inverse_transform * (longitudes_flat, latitudes_flat)

    # Round and convert to integers
    cols = np.floor(cols).astype(int)
    rows = np.floor(rows).astype(int)

    # Get image dimensions
    height, width, _ = image_data.shape

    # Create a mask for valid indices
    valid_mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

    # Prepare an array for the pixel values
    num_pixels = latitudes_flat.size
    num_bands = image_data.shape[-1]
    pixel_values = np.zeros((num_pixels, num_bands), dtype=image_data.dtype)

    # Only retrieve pixel values for valid indices
    if np.any(valid_mask):
        pixel_values[valid_mask] = image_data[rows[valid_mask], cols[valid_mask], :]

    # Handle invalid indices (e.g., set to NaN)
    # pixel_values[~valid_mask] = np.nan  # Uncomment if you prefer NaN for invalid pixels

    # Reshape the output to match the input shape (H x W x bands)
    output_shape = latitudes.shape + (num_bands,)
    pixel_values = pixel_values.reshape(output_shape)

    return pixel_values

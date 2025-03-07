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

    OCEAN_DATA_DIR: ClassVar[str] = os.path.join(__file__, "../ocean_data/")
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
            transform = src.transform

        if image_data.dtype not in GeoTIFFData.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported data type {image_data.dtype}. Supported data types are: "
                f"{', '.join(str(dtype) for dtype in GeoTIFFData.SUPPORTED_DTYPES)}."
            )

        # convert from (channels, height, width) to (height, width, channels)
        image_data = np.moveaxis(image_data, 0, -1)
        inverse_transform = ~transform
        return GeoTIFFData(file_path, image_data, inverse_transform)

    def save(self) -> None:
        """
        Save the contents of this GeoTIFFData object to the underlying file specified by self.image_path.
        Note that this will overwrite any existing file at that location.

        Note that this assumes that self.transform maps to pixel coordinates from the EPSG:4326 coordinate reference
        system, which corresponds to (longitude, latitude) coordinates in degrees using the WGS 84 ellipsoid.
        """
        assert self.dtype in GeoTIFFData.SUPPORTED_DTYPES, (
            f"Unsupported data type {self.dtype}. Supported data types are: "
            f"{', '.join(str(dtype) for dtype in GeoTIFFData.SUPPORTED_DTYPES)}."
        )
        height, width, num_channels = self.image_data.shape

        # convert from (height, width, channels) to (channels, height, width)
        image_data = np.moveaxis(self.image_data, -1, 0)
        inverse_transform = ~self.transform

        metadata = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": num_channels,
            "dtype": self.dtype,
            "crs": GeoTIFFData.EPSG_4326_CRS,
            "transform": inverse_transform,
        }
        with rasterio.open(self.image_path, "w", **metadata) as dst:
            dst.write(image_data)

    @staticmethod
    def load_random_ocean_data() -> "GeoTIFFData":
        """
        Get the GeoTIFFData for a sample ocean data file.

        Returns:
            GeoTIFFData: The GeoTIFFData for the ocean data.
        """
        ocean_data_path = np.random.choice(os.listdir(GeoTIFFData.OCEAN_DATA_DIR))
        ocean_data = GeoTIFFData.load(ocean_data_path)
        return ocean_data

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
        scale_x = width / (max_lon - min_lon)
        scale_y = height / (max_lat - min_lat)

        # maps (lon, lat) to (u, v) (i.e. width, height)
        self.transform = Affine(
            # don't flip the x-axis since increasing u corresponds to increasing longitude
            a=scale_x,
            b=0,
            # choose offset such that min_lon maps to u=0
            c=-min_lon * scale_x,
            d=0,
            # flip the y-axis since increasing v corresponds to decreasing latitude
            e=-scale_y,
            # choose offset such that max_lat maps to v=0
            f=max_lat * scale_y,
        )

    def get_pixel_coordinates(
        self, lat_lon: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the pixel coordinates corresponding to the given latitudes and longitudes.
        A mask is also returned to indicate which output pixel coordinates contain valid data.

        :param lat_lon: A numpy array of shape (..., 2) containing the latitudes and longitudes to query.
        :return: A Tuple containing:
                 - A numpy array of shape lat_lon.shape[:-1] containing the horizontal pixel coordinates, u.
                 - A numpy array of shape lat_lon.shape[:-1] containing the vertical pixel coordinates, v.
                 - A numpy array of shape lat_lon.shape[:-1] indicating which pixel coordinates contain valid data.
        """
        assert lat_lon.shape[-1] == 2, "lat_lon must have shape (..., 2)."

        shape_prefix = lat_lon.shape[:-1]
        lat_flat, lon_flat = lat_lon.reshape(-1, 2).T

        us, vs = self.transform * (lon_flat, lat_flat)
        us = np.floor(us).astype(int).reshape(shape_prefix)
        vs = np.floor(vs).astype(int).reshape(shape_prefix)

        height, width, _ = self.image_data.shape
        valid_mask = (vs >= 0) & (vs < height) & (us >= 0) & (us < width)
        return us, vs, valid_mask

    def query_pixel_colors(self, lat_lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query pixel colors from this GeoTIFFData for a set of latitudes and longitudes.

        The pixel colors' channels will be returned in the same order as the GeoTIFF data, which should be in the order
        (red, green, blue).

        :param lat_lon: A numpy array of shape (..., 2) containing the latitudes and longitudes to query.
        :return: A Tuple containing:
                 - A numpy array of shape lat_lon.shape[:-1] + (self.num_channels,) containing the pixel values.
                 - A numpy array of shape lat_lon.shape[:-1] indicating which output pixels contain valid data.
        """
        us, vs, valid_mask = self.get_pixel_coordinates(lat_lon)

        image_flat = np.zeros(
            lat_lon.shape[:-1] + (self.num_channels,), dtype=self.image_data.dtype
        )
        image_flat[valid_mask, :] = self.image_data[vs[valid_mask], us[valid_mask], :]

        return image_flat, valid_mask


class GeoTIFFCache:
    """
    This class is responsible for loading and caching GeoTIFF data for Earth image simulation.

    Attributes:
        FALLBACK_GEOTIFF_FOLDER: Default folder containing GeoTIFF files. Only used if the user configuration file is not found.
    """

    FALLBACK_GEOTIFF_FOLDER = "/home/argus/eedl_images/"

    def __init__(
        self,
        geotiff_folder: str | None = None,
        use_ocean_imagery: bool = True,
        max_cache_size: int | None = 58,
    ):
        """
        Initialize the GeoTIFF cache.

        Parameters:
            geotiff_folder: Path to the folder containing GeoTIFF files.
            use_ocean_imagery: Whether to return GeoTiIFFData objects containing ocean data for regions without data.
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
        self.use_ocean_imagery = use_ocean_imagery

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
    def validate_salient_region_data_exists(geotiff_folder: str) -> None:
        """
        Check if all salient region folders exist in the specified GeoTIFF folder and are not empty.

        Parameters:
            geotiff_folder: Path to the folder containing GeoTIFF files.

        Raises:
            FileNotFoundError: If one or more region folders are not found or are empty.
        """
        salient_region_ids = load_config()["vision"]["salient_mgrs_region_ids"]

        all_regions_have_data = True
        for region in salient_region_ids:
            region_folder = os.path.join(geotiff_folder, region)
            if not os.path.exists(region_folder):
                print(f"WARNING: Region folder '{region_folder}' not found.")
                all_regions_have_data = False
            if len(os.listdir(region_folder)) == 0:
                print(f"WARNING: Region folder '{region_folder}' is empty.")
                all_regions_have_data = False
        if all_regions_have_data:
            print("All salient region folders found and contain data.")
        else:
            raise FileNotFoundError("One or more region folders not found or empty.")

    def load_geotiff_data(self, region: str) -> GeoTIFFData | None:
        """
        Load GeoTIFF data for a specific region.

        Note that this function is dynamically wrapped with an LRU cache in the constructor, so it will cache its
        outputs for recent regions. This makes it likely that temporally adjacent images will be loaded from the cache,
        resulting in consistent image appearance.

        :param region: The MGRS region to load data for.
        :return: A GeoTIFFData object, or None if there is no data for the specified region.
        """

        # TODO: the lru_cache for load_geotiff_data may contain duplicate image data data for different ocean regions
        #       (although they'll have different transforms). We probably want to avoid this, possibly by using a
        #       custom cache implementation.
        def fallback_loader() -> GeoTIFFData | None:
            if not self.use_ocean_imagery:
                return None
            ocean_data = GeoTIFFData.load_random_ocean_data()
            ocean_data.remap_to_mgrs_region(region)
            return ocean_data

        region_folder = os.path.join(self.geotiff_folder, region)
        if not os.path.exists(region_folder):
            return fallback_loader()
        region_files = os.listdir(region_folder)
        if len(region_files) == 0:
            return fallback_loader()

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

"""
Module to simulate and visualize Earth images from satellite data.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import ClassVar, Tuple

import numpy as np
import rasterio
from affine import Affine
from brahe import R_EARTH
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import label

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional GPU dependency
    cp = None

try:
    from cupyx.scipy.ndimage import label as cp_label
except Exception:  # pragma: no cover - optional cupyx dependency
    cp_label = None

from image_simulation.blue_marble_simulator import query_blue_marble_pixel_colors
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
        transform: The affine transformation matrix for the GeoTIFF file which maps a tuple of (latitudes, longitudes)
                   to a tuple of (us, vs) (i.e. pixel coordinates).
    """

    SWAP_INPUTS_AFFINE: ClassVar[Affine] = Affine(a=0, b=1, c=0, d=1, e=0, f=0)
    SUPPORTED_DTYPES: ClassVar[Tuple[type, ...]] = (np.uint8, np.float32)
    EPSG_4326_CRS: ClassVar[CRS] = CRS.from_epsg(4326)

    image_path: str
    image_data: np.ndarray
    transform: Affine
    _gpu_image_data: object = field(default=None, init=False, repr=False)

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
            if src.crs == GeoTIFFData.EPSG_4326_CRS:
                image_data = src.read()
                transform: Affine = src.transform
            else:
                # Reproject non-EPSG:4326 imagery to EPSG:4326 on load so all downstream
                # lat/lon lookup logic can use a consistent coordinate frame.
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs,
                    GeoTIFFData.EPSG_4326_CRS,
                    src.width,
                    src.height,
                    *src.bounds,
                )
                image_data = np.zeros(
                    (src.count, dst_height, dst_width),
                    dtype=src.dtypes[0],
                )

                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=image_data[band_idx - 1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=GeoTIFFData.EPSG_4326_CRS,
                        resampling=Resampling.bilinear,
                    )

                transform = dst_transform

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

    def query_pixel_colors(
        self,
        lat_lon: np.ndarray,
        use_gpu: bool = False,
        return_gpu: bool = False,
    ):
        """
        Query pixel colors from this GeoTIFFData for a set of latitudes and longitudes.

        The pixel colors' channels will be returned in the same order as the GeoTIFF data, which should be in the order
        (red, green, blue).

        :param lat_lon: A numpy array of shape (..., 2) containing the latitudes and longitudes to query.
        :param use_gpu: If True and CuPy is available, perform pixel lookup on GPU.
        :param return_gpu: If True while using GPU, return a CuPy array instead of NumPy.
        :return: Pixel values with shape lat_lon.shape[:-1] + (self.num_channels,).
        """
        if use_gpu and cp is not None:
            return self._query_pixel_colors_gpu(lat_lon, return_gpu=return_gpu)

        us, vs, valid_mask = self.get_pixel_coordinates(lat_lon)

        image = np.zeros(lat_lon.shape[:-1] + (self.num_channels,), dtype=self.image_data.dtype)
        image[valid_mask, :] = self.image_data[vs[valid_mask], us[valid_mask], :]
        return image

    def _query_pixel_colors_gpu(self, lat_lon, return_gpu: bool = False):
        """GPU-accelerated pixel lookup for a batch of lat/lon coordinates."""
        assert cp is not None, "CuPy is required for GPU pixel lookup"
        assert lat_lon.shape[-1] == 2, "lat_lon must have shape (..., 2)."

        shape_prefix = lat_lon.shape[:-1]
        lat_lon_flat = lat_lon.reshape(-1, 2)

        # Affine maps (lat, lon) -> (u, v):
        # u = a*lat + b*lon + c ; v = d*lat + e*lon + f
        a, b, c, d, e, f = self.transform.a, self.transform.b, self.transform.c, self.transform.d, self.transform.e, self.transform.f

        lat_lon_gpu = (
            lat_lon_flat.astype(cp.float32, copy=False)
            if isinstance(lat_lon_flat, cp.ndarray)
            else cp.asarray(lat_lon_flat, dtype=cp.float32)
        )
        lats = lat_lon_gpu[:, 0]
        lons = lat_lon_gpu[:, 1]

        us = cp.rint(a * lats + b * lons + c).astype(cp.int32)
        vs = cp.rint(d * lats + e * lons + f).astype(cp.int32)

        height, width, _ = self.image_data.shape
        us = cp.where(us == width, width - 1, us)
        vs = cp.where(vs == height, height - 1, vs)
        valid = (vs >= 0) & (vs < height) & (us >= 0) & (us < width)

        if self._gpu_image_data is None:
            self._gpu_image_data = cp.asarray(self.image_data)

        out = cp.zeros((lat_lon_flat.shape[0], self.num_channels), dtype=self._gpu_image_data.dtype)
        if bool(cp.any(valid)):
            valid_idx = cp.where(valid)[0]
            out[valid_idx, :] = self._gpu_image_data[vs[valid_idx], us[valid_idx], :]

        out = out.reshape(shape_prefix + (self.num_channels,))
        return out if return_gpu else cp.asnumpy(out)


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
        if not all_regions_have_data:
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
        region_folder = os.path.join(self.geotiff_folder, region)
        if not os.path.exists(region_folder):
            return None
        region_files = os.listdir(region_folder)
        if len(region_files) == 0:
            return None

        selected_file = np.random.choice(region_files)
        file_path = os.path.join(region_folder, selected_file)
        return GeoTIFFData.load(file_path)

    def load_all_geotiff_data(self, region: str) -> list[GeoTIFFData]:
        """Load all GeoTIFF files for a region."""
        region_folder = os.path.join(self.geotiff_folder, region)
        if not os.path.exists(region_folder):
            return []

        geotiffs: list[GeoTIFFData] = []
        for file_name in sorted(os.listdir(region_folder)):
            file_path = os.path.join(region_folder, file_name)
            if not os.path.isfile(file_path):
                continue
            if not file_name.lower().endswith((".tif", ".tiff")):
                continue
            geotiffs.append(GeoTIFFData.load(file_path))

        return geotiffs

    def clear_cache(self) -> None:
        """
        Clear the GeoTIFF cache.
        """
        self.load_geotiff_data.cache_clear()


class EarthImageSimulator:
    """
    Simulator for simulating Earth images from downloaded GeoTIFF files, accounting for satellite position and orientation.
    """

    BLUE_MARBLE_BRIGHTNESS_FACTOR = 2.7

    def __init__(
        self,
        geotiff_cache: GeoTIFFCache | None = None,
        inpaint_blue_marble: bool = True,
        blue_marble_month: str | None = "may",
        use_gpu: bool | None = None,
        preload_regions: list[str] | None = None,
    ):
        """
        Initialize the Earth image simulator.

        Parameters:
            geotiff_cache: The GeoTIFFCache to use. If None, a default GeoTIFFCache will be created.
            inpaint_blue_marble: Whether to inpaint from the Blue Marble dataset for Earth pixels with no valid data.
            blue_marble_month: The month of the Blue Marble dataset to use. A fixed month improves cache locality.
                               If None, a random month is chosen for each image (slower, more disk I/O).
            use_gpu: If True, use CuPy for the heavy ray/intersection math. If None, auto-enable when CuPy is
                     available. If False, force NumPy.
            preload_regions: Optional list of MGRS regions to preload (all tiles) into RAM,
                             and into VRAM when use_gpu is enabled.
        """
        self.cache = geotiff_cache if geotiff_cache is not None else GeoTIFFCache()
        self.inpaint_blue_marble = inpaint_blue_marble
        self.blue_marble_month = blue_marble_month
        if use_gpu is True and cp is None:
            raise ImportError(
                "CuPy is required for GPU image simulation but is not installed in this environment."
            )
        self.use_gpu = bool(cp is not None) if use_gpu is None else bool(use_gpu)

        # Validate that CuPy can actually talk to the CUDA runtime on this host.
        # Some environments have CuPy installed but fail at first runtime call.
        if self.use_gpu and cp is not None:
            try:
                _ = cp.cuda.runtime.getDeviceCount()
                _ = cp.zeros((1,), dtype=cp.float32)
            except Exception as exc:
                if use_gpu is True:
                    raise RuntimeError(
                        "GPU mode was requested, but CuPy CUDA runtime initialization failed. "
                        "Try reinstalling a compatible CuPy build for this environment, or run with --cpu. "
                        f"Original error: {exc}"
                    ) from exc
                print(f"[GPU PROBE] CuPy runtime unavailable, falling back to NumPy. Details: {exc}")
                self.use_gpu = False

        self._gpu_ray_cache: dict[str, object] = {}
        self._preloaded_tiles: dict[str, list[GeoTIFFData]] = {}
        backend = "cupy" if self.use_gpu else "numpy"
        device_msg = ""
        if self.use_gpu and cp is not None:
            try:
                device_msg = f", gpu_count={cp.cuda.runtime.getDeviceCount()}, device={cp.cuda.runtime.getDevice() if cp.cuda.runtime.getDeviceCount() > 0 else 'n/a'}"
            except Exception:
                device_msg = ", gpu_info=unavailable"
        print(f"[GPU PROBE] EarthImageSimulator backend={backend}{device_msg}")

        if preload_regions:
            self.preload_regions(preload_regions)

    def preload_regions(self, regions: list[str]) -> None:
        """Preload all GeoTIFF tiles for selected regions into memory (and VRAM if enabled)."""
        for region in regions:
            tiles = self.cache.load_all_geotiff_data(region)
            if not tiles:
                print(f"[GPU PRELOAD] No tiles found for region={region}")
                continue

            if self.use_gpu and cp is not None:
                for tile in tiles:
                    if tile._gpu_image_data is None:
                        tile._gpu_image_data = cp.asarray(tile.image_data)
                print(f"[GPU PRELOAD] region={region} tiles={len(tiles)} loaded_to=VRAM")
            else:
                print(f"[GPU PRELOAD] region={region} tiles={len(tiles)} loaded_to=RAM")

            self._preloaded_tiles[region] = tiles

    @staticmethod
    def _intersect_ellipsoid_cpu(
        ray_directions: np.ndarray,
        satellite_position: np.ndarray,
        a: float = 6378137.0,
        b: float = 6356752.314245,
    ) -> np.ndarray:
        return intersect_ellipsoid(ray_directions, satellite_position, a=a, b=b)

    @staticmethod
    def _ecef_to_lat_lon_cpu(intersection_points: np.ndarray) -> np.ndarray:
        return ecef_to_lat_lon(intersection_points)

    @staticmethod
    def _intersect_ellipsoid_gpu(
        ray_directions,
        satellite_position,
        a: float = 6378137.0,
        b: float = 6356752.314245,
    ):
        ray_directions_flat = ray_directions.reshape(-1, 3)
        aab_squared = cp.asarray([a, a, b], dtype=ray_directions.dtype) ** 2
        A = cp.sum(ray_directions_flat**2 / aab_squared, axis=1)
        B = 2 * cp.sum(ray_directions_flat * (satellite_position / aab_squared), axis=1)
        C = cp.sum(satellite_position**2 / aab_squared) - 1
        discriminant = B**2 - 4 * A * C

        intersection_points_flat = cp.full_like(ray_directions_flat, cp.nan)
        valid_mask = discriminant >= 0
        if cp.any(valid_mask):
            sqrt_discriminant = cp.sqrt(discriminant[valid_mask])
            t1 = (-B[valid_mask] - sqrt_discriminant) / (2 * A[valid_mask])
            t2 = (-B[valid_mask] + sqrt_discriminant) / (2 * A[valid_mask])
            t = cp.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
            t = cp.where(t > 0, t, cp.nan)
            valid_ray_directions = ray_directions_flat[valid_mask]
            intersection_points_flat[valid_mask] = t[:, None] * valid_ray_directions + satellite_position

        return intersection_points_flat.reshape(ray_directions.shape)

    @staticmethod
    def _ecef_to_lat_lon_gpu(intersection_points):
        shape_prefix = intersection_points.shape[:-1]
        flat = intersection_points.reshape(-1, 3)
        valid_mask = ~cp.isnan(flat).any(axis=1)
        lat_lon_flat = cp.full((flat.shape[0], 2), cp.nan, dtype=flat.dtype)

        if cp.any(valid_mask):
            valid_points = flat[valid_mask]
            x, y, z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]

            lon = cp.degrees(cp.arctan2(y, x))
            e2 = (6378137.0**2 - 6356752.314245**2) / 6378137.0**2
            ep2 = (6378137.0**2 - 6356752.314245**2) / 6356752.314245**2
            p = cp.sqrt(x**2 + y**2)
            theta = cp.arctan2(z * 6378137.0, p * 6356752.314245)
            lat = cp.arctan2(z + ep2 * 6356752.314245 * cp.sin(theta) ** 3,
                             p - e2 * 6378137.0 * cp.cos(theta) ** 3)

            lat_lon_flat[valid_mask, 0] = cp.degrees(lat)
            lat_lon_flat[valid_mask, 1] = lon

        return lat_lon_flat.reshape(*shape_prefix, 2)

    @staticmethod
    def trim_small_connected_components(mask: np.ndarray, min_size: int = 3) -> np.ndarray:
        """
        Remove small connected components from the provided binary mask.

        Parameters:
            mask: A binary mask to trim.
            min_size: The minimum size of connected components to keep.

        Returns:
            The trimmed binary mask.
        """
        assert mask.dtype == bool, "mask must be a binary mask."

        labeled_connected_components, num_labels = label(
            mask, structure=np.ones((3, 3), dtype=bool)
        )

        for label_id in range(1, num_labels + 1):
            connected_component_mask = labeled_connected_components == label_id

            if np.sum(connected_component_mask) < min_size:
                mask[connected_component_mask] = False

        return mask

    @staticmethod
    def _trim_small_connected_components_gpu(mask, min_size: int = 3):
        """GPU connected-component filtering when cupyx is available; CPU fallback otherwise."""
        assert cp is not None, "CuPy is required for GPU mask trimming"
        if cp_label is None:
            trimmed_cpu = EarthImageSimulator.trim_small_connected_components(cp.asnumpy(mask), min_size)
            return cp.asarray(trimmed_cpu)

        labeled_connected_components, num_labels = cp_label(mask, structure=cp.ones((3, 3), dtype=bool))
        for label_id in range(1, int(num_labels) + 1):
            connected_component_mask = labeled_connected_components == label_id
            if int(cp.sum(connected_component_mask).item()) < min_size:
                mask[connected_component_mask] = False
        return mask

    def simulate_image_for_training(
        self,
        position_ecef: np.ndarray,
        ecef_R_body: np.ndarray,
        camera_model: CameraModel,
        return_lat_lon: bool = True,
    ) -> Tuple[Frame, np.ndarray]:
        """
        Simulate an Earth image given the satellite position, attitude, and camera model.
        This method also returns the latitudes and longitudes for each pixel.

        Parameters:
            position_ecef: A numpy array of shape (3,) representing the satellite position in ECEF coordinates.
            ecef_R_body: A numpy array of shape (3, 3) representing the rotation matrix from body to ECEF coordinates.
            camera_model: The camera model to use to simulate the image.

        Returns:
            A Tuple containing:
            - The simulated Frame object.
            - A numpy array of shape CameraModel.RESOLUTION + (2,) containing the latitudes and longitudes for each
              pixel, or np.nan if the pixel does not correspond to any MGRS region.
        """
        assert np.linalg.norm(position_ecef) > R_EARTH, "position_ecef must be outside the Earth."

        ray_directions_body = camera_model.ray_directions_body()

        lat_lon_gpu = None
        if self.use_gpu:
            cam_name = camera_model.camera_name
            ray_directions_body_gpu = self._gpu_ray_cache.get(cam_name)
            if ray_directions_body_gpu is None:
                ray_directions_body_gpu = cp.asarray(ray_directions_body, dtype=cp.float32)
                self._gpu_ray_cache[cam_name] = ray_directions_body_gpu

            ecef_R_body_gpu = cp.asarray(ecef_R_body, dtype=cp.float32)
            ray_directions_ecef = ray_directions_body_gpu @ ecef_R_body_gpu.T
            camera_position_ecef = cp.asarray(
                camera_model.get_camera_position(position_ecef, ecef_R_body), dtype=cp.float32
            )
            intersection_points = EarthImageSimulator._intersect_ellipsoid_gpu(
                ray_directions_ecef, camera_position_ecef
            )
            lat_lon_gpu = EarthImageSimulator._ecef_to_lat_lon_gpu(intersection_points).astype(cp.float32)
            lat_lon = cp.asnumpy(lat_lon_gpu) if return_lat_lon else None
        else:
            ray_directions_ecef = ray_directions_body @ ecef_R_body.T
            camera_position_ecef = camera_model.get_camera_position(position_ecef, ecef_R_body)
            intersection_points = EarthImageSimulator._intersect_ellipsoid_cpu(
                ray_directions_ecef, camera_position_ecef
            )
            lat_lon = EarthImageSimulator._ecef_to_lat_lon_cpu(intersection_points)

        image = cp.zeros(CameraModel.OUTPUT_SHAPE, dtype=CameraModel.DTYPE) if self.use_gpu else np.zeros(
            CameraModel.OUTPUT_SHAPE, dtype=CameraModel.DTYPE
        )

        # Fast path for one-region workflows: skip CPU-heavy per-pixel MGRS labeling.
        # Sampling the preloaded region tile directly is sufficient because out-of-bounds
        # pixels naturally remain zero via GeoTIFF bounds checks.
        single_preloaded_region = (
            next(iter(self._preloaded_tiles.keys()))
            if len(self._preloaded_tiles) == 1 and len(next(iter(self._preloaded_tiles.values()))) > 0
            else None
        )

        if single_preloaded_region is not None:
            preloaded_tiles = self._preloaded_tiles[single_preloaded_region]
            geotiff_data = preloaded_tiles[np.random.randint(0, len(preloaded_tiles))]
            assert geotiff_data.num_channels == CameraModel.NUM_CHANNELS, (
                f"The GeoTIFF data located at '{geotiff_data.image_path}' does not have {CameraModel.NUM_CHANNELS} "
                f"channels as expected in the camera model."
            )
            assert geotiff_data.dtype == CameraModel.DTYPE, (
                f"The GeoTIFF data located at {geotiff_data.image_path} does not have a dtype of "
                f"{CameraModel.DTYPE} as expected in the camera model."
            )
            if self.use_gpu and lat_lon_gpu is not None:
                image = geotiff_data.query_pixel_colors(lat_lon_gpu, use_gpu=True, return_gpu=True)
            else:
                image = geotiff_data.query_pixel_colors(lat_lon, use_gpu=False)
        else:
            # General multi-region path.
            lat_lon_for_regions = cp.asnumpy(lat_lon_gpu) if (self.use_gpu and lat_lon is None) else lat_lon
            mgrs_regions = calculate_mgrs_zones(lat_lon_for_regions)
            present_regions = np.unique(mgrs_regions[mgrs_regions != b""])

            for region in present_regions:
                region_name = str(region, encoding="ascii")
                preloaded = self._preloaded_tiles.get(region_name)
                if preloaded:
                    geotiff_data = preloaded[np.random.randint(0, len(preloaded))]
                else:
                    geotiff_data = self.cache.load_geotiff_data(region_name)
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
                if self.use_gpu and lat_lon_gpu is not None:
                    region_mask_gpu = cp.asarray(region_mask)
                    image[region_mask_gpu, :] = geotiff_data.query_pixel_colors(
                        lat_lon_gpu[region_mask_gpu], use_gpu=True, return_gpu=True
                    )
                else:
                    image[region_mask, :] = geotiff_data.query_pixel_colors(
                        lat_lon_for_regions[region_mask], use_gpu=False
                    )

        if self.inpaint_blue_marble:
            if self.use_gpu and lat_lon_gpu is not None:
                inpaint_mask = (~cp.any(cp.isnan(lat_lon_gpu), axis=-1)) & cp.all(image == 0, axis=-1)

                # Keep morphology on device when cupyx is available.
                inpaint_mask = EarthImageSimulator._trim_small_connected_components_gpu(inpaint_mask)

                if bool(cp.any(inpaint_mask)):
                    inpaint_mask_np = cp.asnumpy(inpaint_mask)
                    lat_lon_for_inpaint = cp.asnumpy(lat_lon_gpu[inpaint_mask])
                    blue_marble_pixel_values = query_blue_marble_pixel_colors(
                        lat_lon_for_inpaint, self.blue_marble_month
                    )
                    blue_marble_pixel_values = np.clip(
                        np.rint(
                            EarthImageSimulator.BLUE_MARBLE_BRIGHTNESS_FACTOR * blue_marble_pixel_values
                        ),
                        0,
                        255,
                    )
                    image[inpaint_mask] = cp.asarray(blue_marble_pixel_values, dtype=image.dtype)
            else:
                inpaint_mask = ~np.any(np.isnan(lat_lon), axis=-1) & np.all(image == 0, axis=-1)

                # avoid inpainting very small connected components of pixels since we want to avoid overwriting data that
                # just happens to consist of zeros by chance, despite being valid data
                inpaint_mask = EarthImageSimulator.trim_small_connected_components(inpaint_mask)

                if np.any(inpaint_mask):
                    blue_marble_pixel_values = query_blue_marble_pixel_colors(
                        lat_lon[inpaint_mask, :], self.blue_marble_month
                    )
                    blue_marble_pixel_values = np.clip(
                        np.rint(
                            EarthImageSimulator.BLUE_MARBLE_BRIGHTNESS_FACTOR * blue_marble_pixel_values
                        ),
                        0,
                        255,
                    )
                    image[inpaint_mask, :] = blue_marble_pixel_values

        if self.use_gpu:
            image = cp.asnumpy(image)

        return (
            Frame(image, camera_model.camera_name, datetime.now()),
            (
                cp.asnumpy(lat_lon_gpu)
                if (self.use_gpu and return_lat_lon and lat_lon_gpu is not None)
                else (lat_lon if return_lat_lon else np.empty((0, 2), dtype=np.float32))
            ),
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
        frame, _ = self.simulate_image_for_training(position_ecef, ecef_R_body, camera_model)
        return frame

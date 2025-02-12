"""
Module to simulate and visualize Earth images from satellite data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from utils.config_utils import load_config

# pylint: disable=import-error
from utils.earth_utils import (
    calculate_mgrs_zones,
    ecef_to_lat_lon,
    get_nadir_rotation,
    lat_lon_to_ecef,
)


class EarthImageSimulator:
    def __init__(self, geotiff_folder=None, resolution=None, hfov=None):
        """
        Initialize the Earth image simulator.

        Parameters:
            geotiff_folder (str): Path to the folder containing GeoTIFF files.
            resolution (tuple): Camera resolution (width, height).
            hfov (float): Horizontal field of view in degrees.
        """
        if geotiff_folder is None:
            geotiff_folder = "/home/argus/eedl_images/"
        if resolution is None:
            resolution = np.array([4608, 2592])  # width, height
        if hfov is None:
            hfov = 66.1
        self.cache = GeoTIFFCache(geotiff_folder)
        self.resolution = resolution
        self.camera = CameraSimulation(self.resolution, hfov)

    def simulate_image(self, position, orientation):
        """
        Simulate an Earth image given the satellite position and orientation.

        Parameters:
            position (np.ndarray): Satellite position in ECEF coordinates (3,).
            orientation (np.ndarray): Satellite orientation as a 3x3 rotation matrix from the camera frame to ECEF.

        Returns:
            np.ndarray: Simulated RGB image.
        """
        # Generate ray directions in ECEF frame
        ray_directions_ecef = self.camera.rays_in_ecef(orientation)

        # Intersect rays with the Earth
        intersection_points = intersect_ellipsoid(ray_directions_ecef, position)

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
        width, height = self.resolution
        pixel_colors_full = np.zeros((height, width, 3), dtype=np.uint8)

        # Load and assign data for each region
        for region in present_regions:
            data, trans = self.cache.load_geotiff_data(region)
            if data is None:
                continue

            # Mask for the current region
            region_mask = (mgrs_regions == region).reshape(height, width)

            # Skip if no pixels belong to this region
            if not np.any(region_mask):
                continue

            # Query pixel colors for the region
            pixel_colors_region = query_pixel_colors(
                latitudes[region_mask.flatten()], longitudes[region_mask.flatten()], data, trans
            )

            # Assign pixel values to the full image
            pixel_colors_full[region_mask] = pixel_colors_region

        return pixel_colors_full

    def display_image(self, image):
        """
        Display the simulated image.

        Parameters:
            image (np.ndarray): Simulated RGB image.
        """
        plt.imshow(image)
        plt.axis("off")
        plt.show()


class GeoTIFFCache:
    def __init__(self, geotiff_folder: str):
        """
        Initialize the GeoTIFF cache.

        Parameters: geotiff_folder (str): Path to the folder containing GeoTIFF files.
        """
        self.geotiff_folder = geotiff_folder
        self.cache = {}

        for region in [
            "10S",
            "10T",
            "11R",
            "12R",
            "16T",
            "17R",
            "17T",
            "18S",
            "32S",
            "32T",
            "33S",
            "33T",
            "52S",
            "53S",
            "54S",
            "54T",
        ]:
            region_folder = os.path.join(self.geotiff_folder, region)
            if not os.path.exists(region_folder):
                print(f"WARNING: Region folder '{region_folder}' not found.")
                break
        else:
            print("All region folders found!")

    def load_geotiff_data(self, region):
        if region in self.cache:
            return self.cache[region]

        region_folder = os.path.join(self.geotiff_folder, region)
        if not os.path.exists(region_folder):
            self.cache[region] = (None, None)
            return self.cache[region]
        region_files = os.listdir(region_folder)
        if not region_files:
            self.cache[region] = (None, None)
            return self.cache[region]

        selected_file = np.random.choice(region_files)
        file_path = os.path.join(region_folder, selected_file)
        with rasterio.open(file_path) as src:
            data = src.read()
            data = np.moveaxis(data, 0, -1)
            trans = src.transform
        self.cache[region] = (data, trans)
        return self.cache[region]

    def clear_cache(self):
        self.cache = {}


class CameraSimulation:
    def __init__(self, resolution, fov):
        """
        Initialize the simulation camera parameters

        Parameters:
            resolution (tuple): Resolution of the camera (width, height).
            fov (float): Field of view in degrees (assumes square FOV).
        """
        self.resolution = resolution
        self.fov = np.radians(fov)  # Convert FOV to radians

    def ray_directions(self):
        """
        Generate ray directions for the camera.

        Returns:
            np.ndarray: Array of ray directions (HxWx3) in the camera frame.
        """
        width, height = self.resolution
        half_width = np.tan(self.fov / 2)
        half_height = half_width * (height / width)

        x = np.linspace(-half_width, half_width, width)
        y = np.linspace(-half_height, half_height, height)
        xx, yy = np.meshgrid(x, y)
        zz = np.ones_like(xx)  # Assume unit depth

        # Stack and normalize ray directions
        ray_directions = np.stack([xx, yy, zz], axis=-1)
        ray_directions /= np.linalg.norm(ray_directions, axis=-1, keepdims=True)
        return ray_directions

    def rays_in_ecef(self, orientation):
        """
        Transform ray directions from the camera frame to the ECEF frame.

        Parameters:
            orientation (np.ndarray): 3x3 rotation matrix for orientation.

        Returns:
            np.ndarray: Array of ray directions (HxWx3) in the ECEF frame.
        """
        return self.ray_directions() @ orientation.T

    def pixel_to_bearing_unit_vector(self, pixel_coords):
        """
        Converts pixel coordinates to bearing unit vectors in the camera frame.

        Parameters:
            pixel_coords (np.ndarray): An array of shape (N, 2) with pixel coordinates.

        Returns:
            np.ndarray: An array of shape (N, 3) with bearing unit vectors in the camera frame.
        """
        width, height = self.resolution

        half_width = np.tan(self.fov / 2)
        half_height = half_width * (height / width)

        u = pixel_coords[:, 0]  # Pixel x-coordinates
        v = pixel_coords[:, 1]  # Pixel y-coordinates

        # Normalize pixel coordinates to range [-half_width, half_width] and [half_height, -half_height]
        # Assuming pixel (0,0) is at the top-left corner
        x = -half_width + (2 * half_width) * (u / (width - 1))
        y = half_height - (2 * half_height) * (
            v / (height - 1)
        )  # Invert y-axis for image coordinates
        z = np.ones_like(x)

        # Stack and normalize direction vectors
        bearing_unit_vectors = np.stack([x, y, z], axis=-1)
        bearing_unit_vectors /= np.linalg.norm(bearing_unit_vectors, axis=1, keepdims=True)
        return bearing_unit_vectors


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


def sweep_lat_lon_test():
    config = load_config()
    body_R_camera = np.asarray(config["satellite"]["camera"]["body_R_camera"])
    simulator = EarthImageSimulator()

    latitudes = np.linspace(-90, 90, 90)
    longitudes = np.linspace(-180, 180, 90)

    lat_lon = np.stack(np.meshgrid(latitudes, longitudes), axis=-1)
    ecef_positions = lat_lon_to_ecef(lat_lon)

    # scale to 600km altitude
    R_earth = 6371.0088e3
    ecef_positions *= (R_earth + 600e3) / R_earth

    i_stride = ecef_positions.shape[1]
    total = np.prod(ecef_positions.shape[:2])
    empty_indices = []
    for i, j in np.ndindex(ecef_positions.shape[:2]):
        ecef_position = ecef_positions[i, j, :]
        ecef_velocity = np.array([0, 0, 1])

        orientation = get_nadir_rotation(np.concatenate((ecef_position, ecef_velocity)))
        simulated_image = simulator.simulate_image(ecef_position, orientation @ body_R_camera)

        if j % 20 == 0:
            print(f"{i * i_stride + j}/{total}")
        if np.all(simulated_image == 0):
            empty_indices.append((i, j))
        else:
            print(f"Nonempty image at index ({i}, {j}), lat/lon: {lat_lon[i, j, :]}")

    print(f"{len(empty_indices)}/{total} images are empty")
    print(f"Empty images at indices: {empty_indices}")

    with open("empty_images.txt", "w") as f:
        f.write(str(empty_indices))


def main():
    config = load_config()
    body_R_camera = np.asarray(config["satellite"]["camera"]["body_R_camera"])
    simulator = EarthImageSimulator()

    lat_lon = np.array([39.8283, -98.5795])
    ecef_position = lat_lon_to_ecef(lat_lon[np.newaxis, np.newaxis, :])[0, 0, :]
    R_earth = 6371.0088e3
    ecef_position *= (R_earth + 6000e3) / np.linalg.norm(ecef_position)
    ecef_velocity = np.array([0, 0, 1])
    orientation = get_nadir_rotation(np.concatenate((ecef_position, ecef_velocity)))

    simulated_image = simulator.simulate_image(ecef_position, orientation @ body_R_camera)
    print(np.all(simulated_image == 0))


if __name__ == "__main__":
    # sweep_lat_lon_test()
    main()

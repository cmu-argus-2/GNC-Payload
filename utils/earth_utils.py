"""
Common earth utilities.
"""

import numpy as np


# TODO: use brahe constants instead of hardcoding
def convert_to_lat_lon(intersection_points, a=6378137.0, b=6356752.314245):
    """
    Convert intersection points (ECEF) to latitude and longitude.

    Parameters:
        intersection_points (np.ndarray): Array of intersection points (HxWx3) in ECEF coordinates.

    Returns:
        np.ndarray: Array of latitude and longitude (HxWx2), or NaN for invalid points.
    """
    # TODO: generalize this to work with arbitrary arrays of shape (..., 2)
    H, W, _ = intersection_points.shape
    intersection_points_flat = intersection_points.reshape(-1, 3)

    valid_mask = ~np.isnan(intersection_points_flat).any(axis=1)

    lat_lon_flat = np.full((H * W, 2), np.nan)

    valid_points = intersection_points_flat[valid_mask]

    x, y, z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]

    # Longitude calculation (same for geodetic and geocentric)
    lon = np.degrees(np.arctan2(y, x))

    # Geodetic latitude calculation (iterative approach)
    e2 = (a**2 - b**2) / a**2  # First eccentricity squared
    ep2 = (a**2 - b**2) / b**2  # Second eccentricity squared
    p = np.sqrt(x**2 + y**2)

    # Initial approximation of latitude
    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3)

    # Convert to degrees
    lat = np.degrees(lat)

    # Store results in flat array
    lat_lon_flat[valid_mask, 0] = lat
    lat_lon_flat[valid_mask, 1] = lon

    return lat_lon_flat.reshape(H, W, 2)


def lat_lon_to_ecef(lat_lon, a=6378137.0, b=6356752.314245):
    """
    Convert latitude and longitude to ECEF (Earth-Centered, Earth-Fixed) coordinates.

    Parameters:
        lat_lon (np.ndarray): Array of latitude and longitude (HxWx2).

    Returns:
        np.ndarray: Array of ECEF coordinates (HxWx3).
    """
    # TODO: generalize this to work with arbitrary arrays of shape (..., 2)
    # H, W = lat_lon.shape
    lat_lon_flat = lat_lon.reshape(-1, 2)

    lat = lat_lon_flat[:, 0]
    lon = lat_lon_flat[:, 1]

    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # First eccentricity squared
    e2 = (a**2 - b**2) / a**2

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    # Assume height h = 0 (on the ellipsoid)
    x = N * np.cos(lat_rad) * np.cos(lon_rad)
    y = N * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2)) * np.sin(lat_rad)

    ecef_flat = np.column_stack((x, y, z))
    # ecef = ecef_flat.reshape(H, W, 3)

    return ecef_flat


def get_nadir_rotation(state: np.ndarray) -> np.ndarray:
    """
    Get the rotation matrix that points the +x camera towards the center of the Earth.
    The body z-axis will point along the orbital angular momentum vector, the body x-axis will point towards the
    center of the Earth, and the body y-axis will complete the right-handed orthonormal basis.

    This function is agnostic to the frame of reference of the input state. The output rotation matrix
    will be from the frame of reference of the input state to the body frame.

    Parameters:
        state: A numpy array of shape (6,) containing the position and velocity of the satellite.

    Returns:
        A numpy array of shape (3, 3) representing the rotation matrix from the input state frame to the body frame.
    """
    pos, vel = state[:3], state[3:]
    angular_momentum_dir = np.cross(pos, vel)

    x_plus_dir = -pos / np.linalg.norm(pos)
    z_plus_dir = angular_momentum_dir / np.linalg.norm(angular_momentum_dir)
    y_plus_dir = np.cross(z_plus_dir, x_plus_dir)

    return np.column_stack([x_plus_dir, y_plus_dir, z_plus_dir])


# Define MGRS latitude bands and UTM exceptions
mgrs_latitude_bands = [
    {"name": "C", "min_lat": -80, "max_lat": -72},
    {"name": "D", "min_lat": -72, "max_lat": -64},
    {"name": "E", "min_lat": -64, "max_lat": -56},
    {"name": "F", "min_lat": -56, "max_lat": -48},
    {"name": "G", "min_lat": -48, "max_lat": -40},
    {"name": "H", "min_lat": -40, "max_lat": -32},
    {"name": "J", "min_lat": -32, "max_lat": -24},
    {"name": "K", "min_lat": -24, "max_lat": -16},
    {"name": "L", "min_lat": -16, "max_lat": -8},
    {"name": "M", "min_lat": -8, "max_lat": 0},
    {"name": "N", "min_lat": 0, "max_lat": 8},
    {"name": "P", "min_lat": 8, "max_lat": 16},
    {"name": "Q", "min_lat": 16, "max_lat": 24},
    {"name": "R", "min_lat": 24, "max_lat": 32},
    {"name": "S", "min_lat": 32, "max_lat": 40},
    {"name": "T", "min_lat": 40, "max_lat": 48},
    {"name": "U", "min_lat": 48, "max_lat": 56},
    {"name": "V", "min_lat": 56, "max_lat": 64},
    {"name": "W", "min_lat": 64, "max_lat": 72},
    {"name": "X", "min_lat": 72, "max_lat": 84},  # X spans 12° latitude
]

mgrs_utm_exceptions = [
    {"zone": 32, "min_lon": 3, "max_lon": 12, "bands": ["V"]},  # Norway
    {"zone": 31, "min_lon": 0, "max_lon": 9, "bands": ["X"]},  # Svalbard
    {"zone": 33, "min_lon": 9, "max_lon": 21, "bands": ["X"]},  # Svalbard
    {"zone": 35, "min_lon": 21, "max_lon": 33, "bands": ["X"]},  # Svalbard
    {"zone": 37, "min_lon": 33, "max_lon": 42, "bands": ["X"]},  # Svalbard
]


def calculate_mgrs_zones(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of MGRS regions for given latitude and longitude arrays.

    Parameters:
        latitudes (np.ndarray): 1D or 2D array of latitudes in degrees.
        longitudes (np.ndarray): 1D or 2D array of longitudes in degrees.

    Returns:
        np.ndarray: Array of MGRS region identifiers (same shape as input).
    """
    assert latitudes.shape == longitudes.shape, "latitudes and longitudes must have the same shape"

    # Create lookup tables for vectorized latitude band calculation
    latitude_band_names = np.array([band["name"] for band in mgrs_latitude_bands])
    latitude_band_edges = np.array(
        [[band["min_lat"], band["max_lat"]] for band in mgrs_latitude_bands]
    )

    # Flatten lat/lon for processing
    valid_indices = ~np.isnan(latitudes) & ~np.isnan(longitudes)
    lat_flat = latitudes[valid_indices]
    lon_flat = longitudes[valid_indices]

    # Determine latitude bands
    lat_bands = np.full(lat_flat.shape, None, dtype=object)
    for i, (min_lat, max_lat) in enumerate(latitude_band_edges):
        mask = (lat_flat >= min_lat) & (lat_flat < max_lat)
        lat_bands[mask] = latitude_band_names[i]

    # Determine UTM zones (default calculation)
    utm_zones = ((lon_flat + 180) // 6 + 1).astype(int)

    # Apply UTM exceptions
    for exception in mgrs_utm_exceptions:
        mask = (
            (lon_flat >= exception["min_lon"])
            & (lon_flat < exception["max_lon"])
            & np.isin(lat_bands, exception["bands"])
        )
        utm_zones[mask] = exception["zone"]

    # Combine UTM zones and latitude bands
    mgrs_regions_flat = np.array(
        [f"{zone}{band}" if band is not None else None for zone, band in zip(utm_zones, lat_bands)]
    )

    # Reshape to match input lat/lon shape
    mgrs_regions = np.full(latitudes.shape, None, dtype=object)
    mgrs_regions[valid_indices] = mgrs_regions_flat
    return mgrs_regions

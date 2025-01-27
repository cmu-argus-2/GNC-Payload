"""
Common earth utilities.
"""

import numpy as np

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
    e2 = (a ** 2 - b ** 2) / a ** 2  # First eccentricity squared
    ep2 = (a ** 2 - b ** 2) / b ** 2  # Second eccentricity squared
    p = np.sqrt(x ** 2 + y ** 2)

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
    H, W, _ = lat_lon.shape
    lat_lon_flat = lat_lon.reshape(-1, 2)

    lat = lat_lon_flat[:, 0]
    lon = lat_lon_flat[:, 1]

    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # First eccentricity squared
    e2 = (a ** 2 - b ** 2) / a ** 2

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    # Assume height h = 0 (on the ellipsoid)
    x = N * np.cos(lat_rad) * np.cos(lon_rad)
    y = N * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2)) * np.sin(lat_rad)

    ecef_flat = np.column_stack((x, y, z))
    ecef = ecef_flat.reshape(H, W, 3)

    return ecef


def get_nadir_rotation(satellite_position):
    """
    Get the rotation matrix that points the nadir of the satellite to the center of the Earth.

    Parameters:
        satellite_position (np.ndarray): Satellite position in ECEF coordinates.

    Returns:
        np.ndarray (np.ndarray): Rotation matrix of dimension 3x3 that points the nadir of the satellite to the center of the Earth.
    """
    # pointing nadir in world coordinates
    x, y, z = satellite_position
    zc = dir_vector = -np.array([x, y, z]) / np.linalg.norm([x, y, z])
    axis_of_rotation_z = np.cross(np.array([0, 0, 1]), dir_vector)
    rc = axis_of_rotation_z = axis_of_rotation_z / np.linalg.norm(axis_of_rotation_z)
    xc = -rc

    yc = np.cross(rc, zc)
    R = np.stack([xc, yc, zc], axis=-1)
    return R


def calculate_mgrs_zones(latitudes, longitudes):
    """
    Vectorized computation of MGRS regions for given latitude and longitude arrays.

    Parameters:
        latitudes (np.ndarray): 1D or 2D array of latitudes in degrees.
        longitudes (np.ndarray): 1D or 2D array of longitudes in degrees.

    Returns:
        np.ndarray: Array of MGRS region identifiers (same shape as input).
    """
    # Create lookup tables for vectorized latitude band calculation
    latitude_band_names = np.array([band["name"] for band in latitude_bands])
    latitude_band_edges = np.array([[band["min_lat"], band["max_lat"]] for band in latitude_bands])

    # Flatten lat/lon for processing
    lat_flat = latitudes.ravel()
    lon_flat = longitudes.ravel()

    # Determine latitude bands
    lat_bands = np.full(lat_flat.shape, None, dtype=object)
    for i, (min_lat, max_lat) in enumerate(latitude_band_edges):
        mask = (lat_flat >= min_lat) & (lat_flat < max_lat)
        lat_bands[mask] = latitude_band_names[i]

    # Determine UTM zones (default calculation)
    utm_zones = ((lon_flat + 180) // 6 + 1).astype(int)

    # Apply UTM exceptions
    for exception in utm_exceptions:
        mask = (
                (lon_flat >= exception["min_lon"]) &
                (lon_flat < exception["max_lon"]) &
                np.isin(lat_bands, exception["bands"])
        )
        utm_zones[mask] = exception["zone"]

    # Combine UTM zones and latitude bands
    mgrs_regions = np.array([f"{zone}{band}" if band is not None else None
                             for zone, band in zip(utm_zones, lat_bands)])

    # Reshape to match input lat/lon shape
    return mgrs_regions.reshape(latitudes.shape)

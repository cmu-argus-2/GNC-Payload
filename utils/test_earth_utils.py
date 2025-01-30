import numpy as np

from utils.earth_utils import lat_lon_to_ecef, ecef_to_lat_lon


def test_geodetic_conversion():
    # lat_lon_to_ecef was ChatGPT generated, it also produced this test

    # Generate a grid of latitude and longitude values
    latitudes = np.linspace(-90, 90, num=10)
    longitudes = np.linspace(-180, 180, num=10)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    H, W = lat_grid.shape
    lat_lon = np.stack((lat_grid, lon_grid), axis=2)  # Shape (H, W, 2)

    # Convert lat/lon to ECEF using the inverse function
    ecef_points = lat_lon_to_ecef(lat_lon)

    # Convert ECEF back to lat/lon using the original function
    lat_lon_reconstructed = ecef_to_lat_lon(ecef_points)

    # Compute differences
    lat_diff = lat_lon[:, :, 0] - lat_lon_reconstructed[:, :, 0]
    lon_diff = lat_lon[:, :, 1] - lat_lon_reconstructed[:, :, 1]

    # Print maximum differences
    print("Maximum latitude difference (degrees):", np.max(np.abs(lat_diff)))
    print("Maximum longitude difference (degrees):", np.max(np.abs(lon_diff)))


if __name__ == "__main__":
    test_geodetic_conversion()

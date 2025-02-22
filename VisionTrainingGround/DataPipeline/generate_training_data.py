import numpy as np
from brahe.constants import R_EARTH
from scipy.spatial.transform import Rotation

from image_simulation.earth_vis import EarthImageSimulator, GeoTIFFCache
from sensors.camera_model import CameraModelManager
from utils.config_utils import load_config
from utils.earth_utils import get_MGRS_grid, get_nadir_rotation, lat_lon_to_ecef

IMAGES_PER_REGION = 1000
NOMINAL_ALTITUDE = 510e3
ALTITUDE_VARIATION = 20e3
OFF_NADIR_VARIATION = np.deg2rad(10)


def main():
    config = load_config()
    image_simulator = EarthImageSimulator(GeoTIFFCache(max_cache_size=0))
    camera_manager = CameraModelManager()
    grid = get_MGRS_grid()
    ecef_velocity = np.array([0, 0, 1])

    for region in config["vision"]["salient_mgrs_region_ids"]:
        min_lon, min_lat, max_lon, max_lat = grid[region]

        for i in range(IMAGES_PER_REGION):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            altitude = NOMINAL_ALTITUDE + np.random.uniform(-ALTITUDE_VARIATION, ALTITUDE_VARIATION)

            ecef_position = lat_lon_to_ecef(lat, lon)
            ecef_position /= (R_EARTH + altitude) / np.linalg.norm(ecef_position)

            perturbed_camera_R_nominal_camera = Rotation.from_euler(
                "ZX", [np.random.uniform(-np.pi, np.pi), np.random.uniform(0, OFF_NADIR_VARIATION)]
            ).as_matrix()
            nominal_body_R_nominal_camera = perturbed_body_R_perturbed_camera = camera_manager["x+"].body_R_camera
            ecef_R_nominal_body = get_nadir_rotation(np.concatenate((ecef_position, ecef_velocity)))
            ecef_R_perturbed_body = (
                ecef_R_nominal_body
                @ nominal_body_R_nominal_camera
                @ perturbed_camera_R_nominal_camera.T
                @ perturbed_body_R_perturbed_camera.T
            )

            frame, mgrs_regions, lat_lon = image_simulator.simulate_image_for_training(
                ecef_position, ecef_R_perturbed_body, camera_manager["x+"]
            )


if __name__ == "__main__":
    main()

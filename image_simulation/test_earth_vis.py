import numpy as np
from brahe.constants import R_EARTH

from image_simulation.earth_vis import EarthImageSimulator
from sensors.camera_model import CameraModelManager
from utils.earth_utils import get_nadir_rotation, lat_lon_to_ecef

CONTIGUOUS_US_CENTER_LAT_LON = np.array([39.8283, -98.5795])
FLORIDA_17R_CENTER_LAT_LON = np.array([32.0, -81.0])


def sweep_lat_lon_test():
    simulator = EarthImageSimulator()
    camera_model_manager = CameraModelManager()

    latitudes = np.linspace(-90, 90, 90)
    longitudes = np.linspace(-180, 180, 90)

    lat_lon = np.stack(np.meshgrid(latitudes, longitudes), axis=-1)
    ecef_positions = lat_lon_to_ecef(lat_lon)
    ecef_positions *= (R_EARTH + 600e3) / R_EARTH

    # indirectly controls the orientation of the image about the nadir axis
    ecef_velocity = np.array([0, 0, 1])

    i_stride = ecef_positions.shape[1]
    total = np.prod(ecef_positions.shape[:2])
    empty_indices = []
    for i, j in np.ndindex(ecef_positions.shape[:2]):
        ecef_position = ecef_positions[i, j, :]
        ecef_R_body = get_nadir_rotation(np.concatenate((ecef_position, ecef_velocity)))
        simulated_image = simulator.simulate_image(
            ecef_position, ecef_R_body, camera_model_manager["x+"]
        ).image

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


def simulate_image(
    lat_lon: np.ndarray = CONTIGUOUS_US_CENTER_LAT_LON,
    altitude: float = 6000e3,
    display_image: bool = True,
) -> None:
    simulator = EarthImageSimulator()
    camera_model_manager = CameraModelManager()

    ecef_position = lat_lon_to_ecef(lat_lon)
    R_earth = 6379.0088e3
    ecef_position *= (R_earth + altitude) / np.linalg.norm(ecef_position)

    # choose a velocity vector that results in north pointing upwards in the image
    ecef_velocity = np.cross(np.array([0, 0, 1]), ecef_position)
    ecef_R_body = get_nadir_rotation(np.concatenate((ecef_position, ecef_velocity)))

    simulated_image = simulator.simulate_image(
        ecef_position, ecef_R_body, camera_model_manager["x+"]
    ).image

    print(f"Simulated image is {'' if np.all(simulated_image == 0) else 'not'} blank")
    if display_image:
        simulator.display_image(simulated_image)


if __name__ == "__main__":
    # sweep_lat_lon_test()
    simulate_image()

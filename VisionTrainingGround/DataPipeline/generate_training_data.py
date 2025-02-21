from image_simulation.earth_vis import EarthImageSimulator
from sensors.camera_model import CameraModelManager
from utils.earth_utils import ecef_to_lat_lon, lat_lon_to_ecef, get_nadir_rotation
from scipy.spatial.transform import Rotation


IMAGES_PER_REGION = 1000
NOMINAL_ALTITUDE = 510e3
ALTITUDE_VARIANCE = 20e3



def main():
    image_simulator = EarthImageSimulator()
    camera_manager = CameraModelManager()



if __name__ == "__main__":
    main()

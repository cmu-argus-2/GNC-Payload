import brahe
from brahe import Epoch
import numpy as np

from dynamics.orbital_dynamics import f
from orbit_utils import get_sso_orbit_state
from utils.brahe_utils import increment_epoch
from utils.earth_utils import ecef_to_lat_lon
from orbit_determination.test_nonlinear_least_squares import load_config
from utils.plotting_utils import plot_ground_track


def test_plot_ground_track():
    config = load_config()

    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] * config["solver"]["world_update_rate"]))
    dt = 1 / config["solver"]["world_update_rate"]
    state = get_sso_orbit_state(starting_epoch, 0, -73, 600e3, northwards=True)

    epoch = starting_epoch
    ecef_positions = np.zeros((N, 3))
    ecef_positions[0, :] = brahe.rECItoECEF(starting_epoch) @ state[:3]
    for i in range(0, N - 1):
        state = f(state, dt)
        epoch = increment_epoch(epoch, dt)
        ecef_positions[i + 1, :] = brahe.rECItoECEF(epoch) @ state[:3]

    lat_lons = ecef_to_lat_lon(ecef_positions[np.newaxis, ...])[0, ...]
    plot_ground_track(lat_lons)


if __name__ == "__main__":
    test_plot_ground_track()

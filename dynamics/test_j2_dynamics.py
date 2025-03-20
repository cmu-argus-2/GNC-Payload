"""
Test the J2 dynamics.
"""

import brahe
import numpy as np
from brahe import Epoch

# pylint: disable=import-error
from dynamics.grav_potential_dynamics import j2_jacobian_auto, j2_jacobian_manual
from utils.config_utils import load_config
from utils.orbit_utils import get_max_sso_latitude, get_sso_orbit_state


def test_j2_dynamics() -> None:
    """
    Test the J2 dynamics.
    """
    config = load_config()
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))

    for _ in range(1000):

        rand_alt = np.random.uniform(100e3, 1000e3)
        max_lat = get_max_sso_latitude(rand_alt)
        rand_lat = np.random.uniform(-max_lat, max_lat)
        rand_lon = np.random.uniform(-180, 180)

        state = get_sso_orbit_state(starting_epoch, rand_lat, rand_lon, rand_alt, northwards=True)
        # state = state / 1e3 # Convert from m to km and m/s to km/s

        auto_diff_result = j2_jacobian_auto(state[:3])
        manual_result = j2_jacobian_manual(state[:3])

        # Check if the results of the two computations are close enough
        assert np.allclose(
            auto_diff_result, manual_result, atol=1e-6
        ), f"Test failed for state: {state}"
        print(f"Test passed for state: {state}")

    print("All tests passed!")


if __name__ == "__main__":
    test_j2_dynamics()

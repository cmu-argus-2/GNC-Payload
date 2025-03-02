"""
Test the J2 dynamics.
"""

import brahe
import numpy as np
from brahe import Epoch

#pylint: disable=import-error
from dynamics.j2_dynamics import j2_derivative, j2_jacobian_auto
from utils.config_utils import load_config
from utils.orbit_utils import get_sso_orbit_state


def test_j2_dynamics() -> None:
    """
    Test the J2 dynamics.
    """
    config = load_config()
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))

    for _ in range(1000):
        rand_lat = np.random.uniform(-90, 90)
        rand_lon = np.random.uniform(-180, 180)
        rand_alt = np.random.uniform(100e3, 1000e3)

        state = get_sso_orbit_state(starting_epoch, rand_lat, rand_lon, rand_alt, northwards=True)

        auto_diff_result = j2_jacobian_auto(state[:3])
        manual_result = j2_derivative(state[:3])
        assert np.allclose(
            auto_diff_result, manual_result, atol=1e-6
        ), f"Test failed for state: {state}"
        print("Test passed for state:", state)


if __name__ == "__main__":
    test_j2_dynamics()

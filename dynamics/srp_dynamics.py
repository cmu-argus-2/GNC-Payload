"""
This module contains the acceleration of the satellite due to solar radiation pressure (SRP)
"""

import brahe
import numpy as np
from brahe.epoch import Epoch
from brahe.orbit_dynamics import srp


def srp_acceleration(r_sat: np.ndarray, area: float, mass: float, epoch: Epoch) -> np.ndarray:
    """
    Computes the acceleration due to solar radiation pressure.

    :param r_sat: state position of satellite
    :param area: cross-sectional area of the satellite
    :param mass: mass of the satellite
    :param epoch: epoch for which to compute the solar radiation pressure

    :return: acceleration due to solar radiation pressure
    """
    r_sun = brahe.ephemerides.sun_position(epc=epoch)
    # Ensure conversions are consistent (km to m and back to km)
    return srp.accel_srp(x=r_sat*1e3, r_sun=r_sun, area=area*1e6, mass=mass) / 1e3  


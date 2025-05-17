"""
Module for computing third body dynamics and their Jacobians
"""

import brahe
import numpy as np
from brahe import GM_MOON, GM_SUN
from brahe.epoch import Epoch
from brahe.orbit_dynamics.gravity import accel_thirdbody_sun
from brahe.orbit_dynamics.gravity import accel_thirdbody_moon

GM_MOON = GM_MOON / 1e9  # km^3/s^2
GM_SUN = GM_SUN / 1e9  # km^3/s^2

def third_body_acceleration(r_sat: np.ndarray, r_body: np.ndarray, mu: float) -> np.ndarray:
    """
    Computes the third body acceleration.

    :param r_sat: state position of satellite
    :param r_body: position of the third body in ECI frame [m]
    :param mu: gravitational parameter of the third body

    :return: third body acceleration
    """
    r = r_sat - r_body
    r_norm = np.linalg.norm(r)

    return -mu * r / (r_norm**3)


def third_body_jacobian(r_sat: np.ndarray, r_body: np.ndarray, mu: float) -> np.ndarray:
    """
    Computes the Jacobian of the third body acceleration.

    :param r_sat: state position of satellite
    :param r_body: position of the third body in ECI frame [m]
    :param mu: gravitational parameter of the third body

    :return: Jacobian of the third body acceleration
    """
    r = r_sat - r_body
    r_norm = np.linalg.norm(r)

    return (mu / r_norm**3) * (np.eye(3) - 3 * np.outer(r, r) / r_norm**2)


def sun_gravity(r_sat: np.ndarray, epoch: Epoch) -> np.ndarray:
    """
    Computes the sun gravity acceleration.

    :param r_sat: state position of satellite
    :param epoch: epoch for which to compute the sun position

    :return: sun gravity acceleration
    """
    return accel_thirdbody_sun(epc=epoch, x_eci=r_sat*1e3) / 1e3


def sun_gravity_jac(r_sat: np.ndarray, epoch: Epoch) -> np.ndarray:
    """
    Computes the Jacobian of the sun gravity acceleration.

    :param r_sat: state position of satellite
    :param epoch: epoch for which to compute the sun position

    :return: Jacobian of the sun gravity acceleration
    """
    r_sun = brahe.ephemerides.sun_position(epc=epoch) / 1e3
    return third_body_jacobian(r_sat=r_sat, r_body=r_sun, mu=GM_SUN)


def moon_gravity(r_sat: np.ndarray, epoch: Epoch) -> np.ndarray:
    """
    Computes the moon gravity acceleration.

    :param r_sat: state position of satellite
    :param epoch: epoch for which to compute the moon position

    :return: moon gravity acceleration
    """
    return accel_thirdbody_moon(epc=epoch, x_eci=r_sat*1e3) / 1e3


def moon_gravity_jac(r_sat: np.ndarray, epoch: Epoch) -> np.ndarray:
    """
    Computes the Jacobian of the moon gravity acceleration.

    :param r_sat: state position of satellite
    :param epoch: epoch for which to compute the moon position

    :return: Jacobian of the moon gravity acceleration
    """
    r_moon = brahe.ephemerides.moon_position(epc=epoch) / 1e3
    return third_body_jacobian(r_sat=r_sat, r_body=r_moon, mu=GM_MOON)

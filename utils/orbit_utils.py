import brahe
import numpy as np
from brahe import Epoch, R_EARTH, GM_EARTH

from utils.earth_utils import lat_lon_to_ecef


def is_over_daytime(epoch: Epoch, cubesat_position: np.ndarray) -> bool:
    """
    Determine if the satellite is above a portion of the Earth that is in daylight.

    :param epoch: The epoch as an instance of brahe's Epoch class.
    :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
    :return: True if the satellite is above the daylight portion of the Earth, False otherwise.
    """
    return np.dot(brahe.ephemerides.sun_position(epoch), cubesat_position) > 0


def get_sso_orbit_state(
    epoch: Epoch, latitude: float, longitude: float, altitude: float, northwards: bool = True
) -> np.ndarray:
    """
    Computes the state vector for a circular sun-synchronous orbit at the given epoch, latitude, longitude, and altitude.

    :param epoch: The epoch at which the satellite is at the specified location and the state vector is computed.
    :param latitude: The latitude of the satellite in degrees.
    :param longitude: The longitude of the satellite in degrees.
    :param altitude: The altitude of the circular orbit in meters.
    :param northwards: If True, then the satellite will be moving northwards at the specified epoch.
                       If False, then the satellite will be moving southwards at the specified epoch.
    :return: A numpy array of shape (6,) containing the state vector of the satellite at the specified epoch,
             which meets the specified conditions.
    """
    if altitude < 0 or altitude > 5973e3:
        # cos_inclination will be less than -1 if altitude > 5973km
        raise ValueError("Altitude must be between 0 and 5973km")
    if np.abs(np.cos(np.deg2rad(latitude))) < 0.001:
        raise ValueError("Latitude must not be too close to the poles")

    a = R_EARTH + altitude
    lat_lon = np.array([latitude, longitude])
    position_ecef = lat_lon_to_ecef(lat_lon[np.newaxis, :])[0,:]
    position_ecef *= a / np.linalg.norm(position_ecef)
    position_eci = brahe.frames.rECItoECEF(epoch).T @ position_ecef

    # https://en.wikipedia.org/wiki/Sun-synchronous_orbit#Technical_details
    cos_inclination = -(
        (a / 12_352e3) ** (7 / 2)
    )  # TODO: define this constant in terms of other constants

    # construct a right-handed orthonormal basis (r_hat, z_perp_hat, west_hat)
    r_hat = position_eci / np.linalg.norm(position_eci)
    z_hat = np.array([0, 0, 1])
    z_perp = z_hat - np.dot(z_hat, r_hat) * r_hat
    z_perp_hat = z_perp / np.linalg.norm(z_perp)
    west_hat = np.cross(r_hat, z_perp_hat)

    """
    The orbital normal vector can be represented in this basis as follows:
    n_hat = alpha * z_perp_hat + beta * west_hat + 0 * r_hat
    To match the inclination condition, we need np.dot(n_hat, z_hat) = cos_inclination.
    Note that z_perp_hat is a linear combination of r_hat and z_hat, and west_hat is perpendicular to both r_hat and z_perp_hat;
    thus, west_hat is perpendicular to z_hat (i.e. np.dot(west_hat, z_hat) = 0).
    Thus, cos_inclination = np.dot(n_hat, z_hat) = alpha * np.dot(z_perp_hat, z_hat).
    """
    alpha = cos_inclination / np.dot(z_perp_hat, z_hat)
    beta = np.sqrt(1 - alpha**2)
    normal_1_hat = alpha * z_perp_hat + beta * west_hat
    normal_2_hat = alpha * z_perp_hat - beta * west_hat

    v_magnitude = np.sqrt(GM_EARTH / a)
    v_1 = v_magnitude * np.cross(normal_1_hat, r_hat)
    v_2 = v_magnitude * np.cross(normal_2_hat, r_hat)
    is_v1_northbound = v_1[2] > 0
    is_v2_northbound = v_2[2] > 0

    assert (
        is_v1_northbound != is_v2_northbound
    ), f"Velocities cannot both be {'north' if is_v1_northbound else 'south'}bound!"
    return np.concatenate((position_eci, v_1 if northwards == is_v1_northbound else v_2))

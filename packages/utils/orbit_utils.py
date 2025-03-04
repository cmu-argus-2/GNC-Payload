import brahe
import numpy as np
from brahe import GM_EARTH, R_EARTH, Epoch
from utils.earth_utils import lat_lon_to_ecef


def is_over_daytime(epoch: Epoch, cubesat_position: np.ndarray) -> bool:
    """
    Determine if the satellite is above a portion of the Earth that is in daylight.

    :param epoch: The epoch as an instance of brahe's Epoch class.
    :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
    :return: True if the satellite is above the daylight portion of the Earth, False otherwise.
    """
    return np.dot(brahe.ephemerides.sun_position(epoch), cubesat_position) > 0


def get_cos_sso_inclination(altitude: float) -> float:
    """
    Compute the cosine of the inclination of a sun-synchronous orbit at the given altitude.

    :param altitude: The altitude of the sun-synchronous orbit, in meters.
    :return: The cosine of the inclination of the sun-synchronous orbit at the given altitude.
             This will be in the range [-1, 0) since sun-synchronous orbits are always retrograde.
    """
    if altitude < 0 or altitude > 5973e3:
        # cos_inclination will be less than -1 if altitude > 5973km
        raise ValueError("Altitude must be between 0 and 5973km")

    a = R_EARTH + altitude
    # https://en.wikipedia.org/wiki/Sun-synchronous_orbit#Technical_details
    # TODO: define this distance constant in terms of other constants
    return -((a / 12_352e3) ** (7 / 2))


def get_max_sso_latitude(altitude: float) -> float:
    """
    Compute the maximum possible latitude for a sun-synchronous orbit at the given altitude.
    Note that the minimum possible latitude is the negative of the maximum possible latitude.

    :param altitude: The altitude of the sun-synchronous orbit, in meters.
    :return: The maximum possible latitude for the given altitude, in degrees. This will be between 0 and 90 degrees.
    """
    inclination = np.rad2deg(np.arccos(get_cos_sso_inclination(altitude)))
    assert (
        inclination > 90
    ), "Inclination must be greater than 90 degrees for an sun-synchronous orbit!"
    return 180 - inclination


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
    # No need to actually run this check, since the check on alpha below will catch this condition
    # assert np.abs(latitude) <= get_max_sso_latitude(altitude), "Latitude is out of range for an SSO orbit!"

    cos_inclination = get_cos_sso_inclination(altitude)

    a = R_EARTH + altitude
    position_ecef = lat_lon_to_ecef(np.array([latitude, longitude]))
    position_ecef *= a / np.linalg.norm(position_ecef)
    position_eci = brahe.frames.rECItoECEF(epoch).T @ position_ecef

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
    if np.abs(alpha) > 1:
        inclination = np.rad2deg(np.arccos(cos_inclination))
        max_sso_latitude = get_max_sso_latitude(altitude)
        raise ValueError(
            f"An SSO orbit at an altitude of {altitude / 1000:.2f}km requires an inclination of {inclination:.2f}"
            f"degrees, so the latitude must be between {-max_sso_latitude:.2f} and {max_sso_latitude:.2f} degrees."
        )

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

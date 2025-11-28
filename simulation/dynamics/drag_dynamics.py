"""
Module that defines drag dynamics and its jacobian.
"""

# pylint: disable=import-error
import numpy as np
from brahe import R_EARTH, Epoch
from utils.earth_utils import density_harris_priester

REF_HEIGHT = 600  # km
NOMINAL_DENSITY = 1e-4  # kg/m^3
R_EARTH = R_EARTH / 1e3  # km

# Exponential model parameters from U.S. Standard Atmosphere 1976
# Taken from Fundamentals of Astrodynamics and Applications, 4th Edition, by David A. Vallado
H_ELLP = [300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 700.0]
NOMINAL_DENSITY = [2.418e-2, 9.518e-3, 3.725e-3, 1.585e-3, 6.967e-4, 1.454e-4]  # kg/km^3
SCALE_HEIGHT = [53.628, 53.298, 58.515, 60.828, 63.822, 71.835]  # km


def drag_dynamics(x: np.ndarray, drag_const: float, latest_epoch: Epoch) -> np.ndarray:
    """
    Computes the drag acceleration.

    :param x: Vector consisting of position and velocity ([m, m/s])
    :param drag_const: Drag constant in m^2/kg, calculated as -0.5 * CD * AREA / MASS
    :param latest_epoch: latest epoch for which to compute the density parameter

    :return: drag acceleration
    """
    v = x[3:6]
    density = density_harris_priester(x=x * 1e3, epoch=latest_epoch)
    v_norm = np.linalg.norm(v)

    if np.isclose(v_norm, 0):
        a_drag = np.zeros(3)
    else:
        a_drag = density * v * drag_const * v_norm

    return a_drag


def drag_jacobian(x: np.ndarray, drag_const: float, latest_epoch: Epoch) -> np.ndarray:
    """
    Compute the drag acceleration jacobian.

    :param x: Vector consisting of position and velocity ([m, m/s])
    :param drag_const: Drag constant in m^2/kg, calculated as -0.5 * CD * AREA / MASS
    :param latest_epoch: latest epoch for which to compute the density parameter

    :return: drag acceleration jacobian
    """

    # Technically we need to take the partial derivative of the density with respect to the position as well
    # since the Harris-Priester model is a function of the position and the velocity.
    # However, since that function is not easily differentiable, we cannot compute the jacobian analytically.
    # Since the groundtruth dynamics doesn't require this function anyways, I would ignore this problem for now.
    v = x[3:6]
    v_norm = np.linalg.norm(v)

    if np.isclose(v_norm, 0):
        return np.zeros((3, 3))

    density = density_harris_priester(x=x * 1e3, epoch=latest_epoch)

    da_drag_dv = density * drag_const * ((np.eye(3) * v_norm) - np.outer(v, v) / v_norm)

    return da_drag_dv


def density_exponential(x: np.ndarray) -> float:
    """
    Compute the density using an exponential model based on the Harris-Priester model.

    :param x: Vector consisting of position ([m])
    :param d_est: Drag scalar estimate term

    :return: density in kg/m^3
    """
    r = np.linalg.norm(x[0:3])  # Position vector norm
    h_ellp = r - R_EARTH  # Height above the ellipsoid in km

    if h_ellp < H_ELLP[0]:
        idx = 0
    elif h_ellp > H_ELLP[-1]:
        idx = len(H_ELLP) - 2
    else:
        idx = max(i for i, h in enumerate(H_ELLP) if h < h_ellp)

    ref_height = H_ELLP[idx]  # Convert to meters
    nominal_density = NOMINAL_DENSITY[idx]
    scale_height = SCALE_HEIGHT[idx]  # in km

    height = h_ellp - ref_height
    density_estimate = nominal_density * np.exp(-height / scale_height)

    return density_estimate


def drag_scalar_estimate(x: np.ndarray, d_est: float, drag_const: float) -> np.ndarray:
    """
    Compute the drag acceleration using a scalar drag estimate.
    The formulation is based on the formulation provided by Montebruc, and Gill in
    Satellite Orbits: Models, Methods, and Applications in Chapter 3.5.1 page 86
    The Upper Atmosphere Model. The model has been simplified to use a constant REF_HEIGHT
    Rather than the density scale height calculation that is provided.

    :param x: Vector consisting of position and velocity ([m, m/s])
    :param dest: Drag scalar estimate term
    :param drag_const: Drag constant in m^2/kg, calculated as -0.5 * CD * AREA / MASS

    :return: drag acceleration
    """
    v = x[3:6]
    v_norm = np.linalg.norm(v)
    drag_a = d_est * density_exponential(x) * v * drag_const * v_norm
    return drag_a


def da_dest_drag_derivative(x: np.ndarray, drag_const: float) -> np.ndarray:
    """
    Compute the derivative of the acceleration dynamics with respect to the drag estimate term.

    :param x: Vector consisting of position and velocity ([m, m/s])
    :param drag_const: Drag constant in m^2/kg, calculated as -0.5 * CD * AREA / MASS

    :return: drag derivative
    """
    v = x[3:6]
    v_norm = np.linalg.norm(v)
    da_drag = density_exponential(x) * v * drag_const * v_norm
    return np.expand_dims(da_drag, axis=1)


def dadrag_dr_partial(x: np.ndarray, d_est: float, drag_const: float) -> np.ndarray:
    """
    Compute the partial derivative of the drag acceleration with respect to position.

    :param x: Vector consisting of position and velocity ([m, m/s])
    :param d_est: Drag scalar estimate term
    :param drag_const: Drag constant in m^2/kg, calculated as -0.5 * CD * AREA / MASS

    :return: drag acceleration partial derivative with respect to position
    """
    r = x[0:3]
    v = x[3:6]
    v_norm = np.linalg.norm(v)
    h_ellp = np.linalg.norm(r) - R_EARTH

    if h_ellp < H_ELLP[0]:
        idx = 0
    elif h_ellp > H_ELLP[-1]:
        idx = len(H_ELLP) - 2
    else:
        idx = max(i for i, h in enumerate(H_ELLP) if h < h_ellp)
    ref_height = H_ELLP[idx]  # Convert to meters
    nominal_density = NOMINAL_DENSITY[idx]
    scale_height = SCALE_HEIGHT[idx]  # in km

    height = h_ellp - ref_height
    density_estimate = d_est * nominal_density * np.exp(-height / scale_height)
    dadrag_dr = (
        density_estimate
        * drag_const
        * v_norm
        * np.outer(v, -r)
        / (np.linalg.norm(r) * scale_height)
    )
    return dadrag_dr


def dadrag_dv_partial(x: np.ndarray, d_est: float, drag_const: float) -> np.ndarray:
    """
    Compute the partial derivative of the drag acceleration with respect to velocity.

    :param x: Vector consisting of position and velocity ([m, m/s])
    :param d_est: Drag scalar estimate term
    :param drag_const: Drag constant in m^2/kg, calculated as -0.5 * CD * AREA / MASS

    :return: drag acceleration partial derivative with respect to velocity
    """
    v = x[3:6]
    v_norm = np.linalg.norm(v)
    dadrag_dv = (
        d_est * density_exponential(x) * drag_const * (np.eye(3) * v_norm - np.outer(v, v) / v_norm)
    )
    return dadrag_dv

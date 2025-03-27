"""
Module that defines drag dynamics and its jacobian.
"""

import numpy as np
from brahe import R_EARTH, Epoch

# pylint: disable=import-error
from utils.earth_utils import density_harris_priester

REF_HEIGHT = 600  # km
NOMINAL_DENSITY = 1e-21  # kg/m^3
R_EARTH = R_EARTH / 1e3  # km


def drag_dynamics(x: np.ndarray, drag_const: float, latest_epoch: Epoch) -> np.ndarray:
    """
    Computes the drag acceleration.

    :param x: Vector consisting of position and velocity ([m, m/s])
    :param drag_const: Drag constant in m^2/kg, calculated as -0.5 * CD * AREA / MASS
    :param latest_epoch: latest epoch for which to compute the density parameter

    :return: drag acceleration
    """

    density = density_harris_priester(x=x*1e3, epoch=latest_epoch)
    v_norm = np.linalg.norm(x[3:6])

    if np.isclose(v_norm, 0):
        a_drag = np.zeros(3)
    else:
        a_drag = density * x[3:6] * drag_const * v_norm

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

    v_norm = np.linalg.norm(x[3:6])

    if np.isclose(v_norm, 0):
        return np.zeros((3, 3))

    density = density_harris_priester(x=x*1e3, epoch=latest_epoch)

    da_drag_dv = density * drag_const * ((np.eye(3) * v_norm) - np.outer(x[3:6], x[3:6]) / v_norm)

    return da_drag_dv


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

    v_norm = np.linalg.norm(x[3:6])
    height = np.linalg.norm(x[0:3]) - R_EARTH
    density_estimate = d_est * NOMINAL_DENSITY * np.exp(-height / REF_HEIGHT)
    drag_a = density_estimate * x[3:6] * drag_const * v_norm
    return drag_a


def da_dest_drag_derivative(x: np.ndarray, drag_const: float) -> np.ndarray:
    """
    Compute the derivative of the acceleration dynamics with respect to the drag estimate term.

    :param x: Vector consisting of position and velocity ([m, m/s])
    :param drag_const: Drag constant in m^2/kg, calculated as -0.5 * CD * AREA / MASS

    :return: drag derivative
    """

    v_norm = np.linalg.norm(x[3:6])
    height = np.linalg.norm(x[0:3]) - R_EARTH
    da_drag = NOMINAL_DENSITY * np.exp(-height / REF_HEIGHT) * x[3:6] * drag_const * v_norm
    return np.expand_dims(da_drag, axis=1)


def dadrag_dr_partial(x: np.ndarray, d_est: float, drag_const: float) -> np.ndarray:
    """
    Compute the partial derivative of the drag acceleration with respect to position.

    :param x: Vector consisting of position and velocity ([m, m/s])
    :param d_est: Drag scalar estimate term
    :param drag_const: Drag constant in m^2/kg, calculated as -0.5 * CD * AREA / MASS

    :return: drag acceleration partial derivative with respect to position
    """
    v_norm = np.linalg.norm(x[3:6])
    height = np.linalg.norm(x[0:3]) - R_EARTH
    density_estimate = d_est * NOMINAL_DENSITY * np.exp(-height / REF_HEIGHT)
    dadrag_dr = (
        density_estimate
        * drag_const
        * v_norm
        * np.outer(x[3:6], -x[0:3])
        / (np.linalg.norm(x[0:3]) * REF_HEIGHT)
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
    v_norm = np.linalg.norm(x[3:6])
    height = np.linalg.norm(x[0:3]) - R_EARTH
    density_estimate = d_est * NOMINAL_DENSITY * np.exp(-height / REF_HEIGHT)
    dadrag_dv = (
        density_estimate * drag_const * (np.eye(3) * v_norm - np.outer(x[3:6], x[3:6]) / v_norm)
    )
    return dadrag_dv

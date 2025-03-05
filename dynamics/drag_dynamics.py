"""
Module that defines drag dynamics and its jacobian.
"""

import numpy as np
from brahe import Epoch

# pylint: disable=import-error
from utils.earth_utils import density_harris_priester


def drag_dynamics(x: np.ndarray, drag_const: float, latest_epoch: Epoch) -> np.ndarray:
    """
    Computes the drag acceleration.

    :param x: state vector
    :param drag_const: Drag constant, calculated as -0.5 * CD * AREA / MASS
    :param latest_epoch: latest epoch for which to compute the density parameter

    :return: drag acceleration
    """

    density = density_harris_priester(x=x, epoch=latest_epoch)
    v_norm = np.linalg.norm(x[3:6])

    if v_norm == 0:
        a_drag = np.zeros(3)
    else:
        a_drag = density * x[3:6] * drag_const / v_norm

    return a_drag


def drag_jacobian(x: np.ndarray, drag_const: float, latest_epoch: Epoch) -> np.ndarray:
    """
    Compute the drag acceleration jacobian.

    :param x: state vector
    :param drag_const: Drag constant, calculated as -0.5 * CD * AREA / MASS
    :param latest_epoch: latest epoch for which to compute the density parameter

    :return: drag acceleration jacobian
    """

    v_norm = np.linalg.norm(x[3:6])
    if v_norm == 0:
        return np.zeros((3, 3))

    density = density_harris_priester(x=x, epoch=latest_epoch)

    da_drag_dv = (
        density * drag_const * ((np.eye(3) / v_norm) - np.outer(x[3:6], x[3:6]) / v_norm**3)
    )

    return da_drag_dv

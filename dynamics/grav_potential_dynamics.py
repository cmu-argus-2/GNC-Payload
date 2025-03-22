"""
J2 dynamics and jacobian
"""

import jax
import jax.numpy as jnp
import numpy as np
from brahe.constants import GM_EARTH, J2_EARTH, R_EARTH

F_DYN = (3 * GM_EARTH * J2_EARTH * R_EARTH**2) / 2
F_DER = 1.5 * J2_EARTH * GM_EARTH * R_EARTH**2

# Values for J3 and J4 taken from Wikipedia
# https://en.wikipedia.org/wiki/Geopotential_spherical_harmonic_model#Available_models
J3 = 2.532435346e-6
J4 = 1.619331205e-6

F_DYN_J3 = (5 * GM_EARTH * R_EARTH**3 * J3) / 2
F_DYN_J4 = (35 * GM_EARTH * R_EARTH**4 * J4) / 8


def j3_dynamics(r: jnp.ndarray) -> jnp.ndarray:
    """
    Provide the J3 perturbation acceleration components.

    The formulas use a common formulation:

        a_x = (F_DYN_J3 / r^7) * x * z * (7*(z/r)^2 - 3)
        a_y = (F_DYN_J3 / r^7) * y * z * (7*(z/r)^2 - 3)
        a_z = (F_DYN_J3 / r^7) * z * (35*(z/r)^4 - 30*(z/r)^2 + 3)

    :param r: position vector [x, y, z]
    :return: acceleration vector due to the J3 perturbation
    """
    r_norm = jnp.linalg.norm(r)
    z_over_r = r[2] / r_norm
    F = F_DYN_J3 / r_norm**7
    a_x = F * r[0] * r[2] * (7 * z_over_r**2 - 3)
    a_y = F * r[1] * r[2] * (7 * z_over_r**2 - 3)
    a_z = F * r[2] * (35 * z_over_r**4 - 30 * z_over_r**2 + 3)
    return jnp.array([a_x, a_y, a_z])


def j3_jacobian_auto(r: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Jacobian of the J3 perturbation dynamics using autodiff.

    :param r: position vector [x, y, z]
    :return: Jacobian matrix of shape (3, 3)
    """
    jac = jax.jacobian(j3_dynamics)(r)
    return jac


def j4_dynamics(r: jnp.ndarray) -> jnp.ndarray:
    """
    Provide the J4 perturbation acceleration components.

    The formulas use a common formulation:

        a_x = (F_DYN_J4 / r^9) * x * [63*(z/r)^4 - 70*(z/r)^2 + 15]
        a_y = (F_DYN_J4 / r^9) * y * [63*(z/r)^4 - 70*(z/r)^2 + 15]
        a_z = (F_DYN_J4 / r^9) * z * [315*(z/r)^4 - 420*(z/r)^2 + 105]

    :param r: position vector [x, y, z]
    :return: acceleration vector due to the J4 perturbation
    """
    r_norm = jnp.linalg.norm(r)
    z_over_r = r[2] / r_norm
    F = F_DYN_J4 / r_norm**9
    poly_xy = 63 * z_over_r**4 - 70 * z_over_r**2 + 15
    poly_z = 315 * z_over_r**4 - 420 * z_over_r**2 + 105
    a_x = F * r[0] * poly_xy
    a_y = F * r[1] * poly_xy
    a_z = F * r[2] * poly_z
    return jnp.array([a_x, a_y, a_z])


def j4_jacobian_auto(r: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Jacobian of the J4 perturbation dynamics using autodiff.

    :param r: position vector [x, y, z]
    :return: Jacobian matrix of shape (3, 3)
    """
    jac = jax.jacobian(j4_dynamics)(r)
    return jac


def j2_dynamics(r: jnp.ndarray) -> jnp.ndarray:
    """
    Provide the J2 perturbation dynamics

    :param r: position vector
    :return: The applied force resulting from J2 perturbation dynamics
    """
    r_norm = jnp.linalg.norm(r)

    F = F_DYN / r_norm**5
    a_x = F * (r[0]) * (5 * (r[2] / r_norm) ** 2 - 1)
    a_y = F * (r[1]) * (5 * (r[2] / r_norm) ** 2 - 1)
    a_z = F * (r[2]) * (5 * (r[2] / r_norm) ** 2 - 3)

    return jnp.array([a_x, a_y, a_z])


def j2_jacobian_auto(r: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Jacobian of the J2 perturbation dynamics using autodiff.

    :param r: position vector [x, y, z]
    :return: Jacobian matrix of shape (3, 3)
    """
    jac = jax.jacobian(j2_dynamics)(r)
    return jac


def j2_jacobian_manual(r: np.ndarray) -> np.ndarray:
    """
    Provide the J2 perturbation jacobian without relying on autodiff
    :param r: position vector

    :return: J2 perturbation jacobian of shape (3, 3)
    """
    r_norm = np.linalg.norm(r)

    #                             [dj2x_dx, dj2x_dy, dj2x_dz]
    #             Jacobian   =    [dj2y_dx, dj2y_dy, dj2y_dz]
    #                             [dj2z_dx, dj2z_dy, dj2z_dz]

    #             Terms in the Jacobian have the following shape.

    # dj2x_dx
    da_dx = F_DER * (r_norm**2 - 5 * r[0] ** 2) / r_norm**7
    db_dx = -10 * r[0] * r[2] ** 2 / r_norm**4
    dj2x_dx = da_dx * ((5 * r[2] ** 2 / (r_norm**2)) - 1) + db_dx * F_DER * r[0] / r_norm**5

    # dj2x_dy
    da_dx = -5 * F_DER * r[0] * r[1] / r_norm**7
    db_dx = -10 * r[1] * r[2] ** 2 / r_norm**4
    dj2x_dy = da_dx * ((5 * r[2] ** 2 / (r_norm**2)) - 1) + db_dx * F_DER * r[0] / r_norm**5

    # dj2x_dz
    da_dx = -5 * F_DER * r[0] * r[2] / r_norm**7
    db_dx = (10 * r[2] * r_norm**2 - (10 * r[2] ** 3 - 2 * r[2])) / r_norm**4
    dj2x_dz = da_dx * ((5 * r[2] ** 2 / (r_norm**2)) - 1) + db_dx * F_DER * r[0] / r_norm**5

    # dj2y_dx
    da_dx = -5 * F_DER * r[1] * r[0] / r_norm**7
    db_dx = -10 * r[0] * r[2] ** 2 / r_norm**4
    dj2y_dx = da_dx * ((5 * r[2] ** 2 / (r_norm**2)) - 1) + db_dx * F_DER * r[1] / r_norm**5

    # dj2y_dy
    da_dx = F_DER * (r_norm**2 - 5 * r[1] ** 2) / r_norm**7
    db_dx = -10 * r[1] * r[2] ** 2 / r_norm**4
    dj2y_dy = da_dx * ((5 * r[2] ** 2 / (r_norm**2)) - 1) + db_dx * F_DER * r[1] / r_norm**5

    # dj2y_dz
    da_dx = -5 * F_DER * r[1] * r[2] / r_norm**7
    db_dx = (10 * r[2] * r_norm**2 - (10 * r[2] ** 3 - 2 * r[2])) / r_norm**4
    dj2y_dz = da_dx * ((5 * r[2] ** 2 / (r_norm**2)) - 1) + db_dx * F_DER * r[1] / r_norm**5

    # dj2z_dx
    da_dx = -5 * F_DER * r[2] * r[0] / r_norm**7
    db_dx = (-10 * r[0] * r[2] ** 2) / r_norm**4
    dj2z_dx = da_dx * ((5 * r[2] ** 2 / (r_norm**2)) - 3) + db_dx * F_DER * r[2] / r_norm**5

    # dj2z_dy
    da_dx = -5 * F_DER * r[2] * r[1] / r_norm**7
    db_dx = (-10 * r[1] * r[2] ** 2) / r_norm**4
    dj2z_dy = da_dx * ((5 * r[2] ** 2 / (r_norm**2)) - 3) + db_dx * F_DER * r[2] / r_norm**5

    # dj2z_dz
    da_dx = F_DER * (r_norm**2 - 5 * r[2] ** 2) / r_norm**7
    db_dx = 10 * r[2] * (r_norm**2 - r[2] ** 2) / r_norm**4
    dj2z_dz = da_dx * ((5 * r[2] ** 2 / (r_norm**2)) - 3) + db_dx * F_DER * r[2] / r_norm**5

    return np.array(
        [[dj2x_dx, dj2x_dy, dj2x_dz], [dj2y_dx, dj2y_dy, dj2y_dz], [dj2z_dx, dj2z_dy, dj2z_dz]]
    )

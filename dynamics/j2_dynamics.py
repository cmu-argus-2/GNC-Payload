"""
J2 dynamics and jacobian
"""

import jax
import jax.numpy as jnp
import numpy as np
from brahe.constants import GM_EARTH, J2_EARTH, R_EARTH

F_DYN = (3 * GM_EARTH * J2_EARTH * R_EARTH**2) / 2
F_DER = 1.5 * J2_EARTH * GM_EARTH * R_EARTH**2


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
    Provide the J2 perturbation jacobian using autodiff

    :param r: position vector

    :return: J2 perturbation jacobian of shape (3, 3)
    """
    jac = jax.jacobian(j2_dynamics)(r)
    return jac


def j2_derivative(r: np.ndarray) -> np.ndarray:
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

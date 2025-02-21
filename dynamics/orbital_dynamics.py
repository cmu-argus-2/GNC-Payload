"""
Functions for implementing orbital position dynamics and its jacobian under just the force of gravity.
J2 perturbations are not included.
"""

from typing import Callable

import numpy as np
from brahe.constants import GM_EARTH, J2_EARTH, R_EARTH

from dynamics.j2_dynamics import j2_derivative, j2_jacobian_auto
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from utils.earth_utils import density_harris_priester


def state_derivative(x: np.ndarray) -> np.ndarray:
    """
    The continuous-time state derivative function, \dot{x} = f_c(x), for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :return: A numpy array of shape (6,) containing the state derivative.
    """
    r = x[:3]
    v = x[3:]

    a = -r * GM_EARTH / np.linalg.norm(r) ** 3

    return np.concatenate([v, a])


def state_derivative_jac(x: np.ndarray) -> np.ndarray:
    """
    The continuous-time state derivative Jacobian function, d(f_c)/dx, for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :return: A numpy array of shape (6, 6) containing the state derivative Jacobian.
    """
    ### Choose between using the autodiff jacobian or the manually derived jacobian.
    # j2da_dv = j2_derivative(x[:3])
    # TODO: RETURN TO BASE JACOBIAN AND FINISH WRAPPER
    j2_auto = j2_jacobian_auto(x[:3])
    r = x[:3]
    r_norm = np.linalg.norm(r)
    dv_dr = np.zeros((3, 3))
    da_dr = (
        (-GM_EARTH / r_norm**3) * np.eye(3) + (3 * GM_EARTH / r_norm**5) * np.outer(r, r) + j2_auto
    )
    dv_dv = np.eye(3)
    da_dv = np.zeros((3, 3))
    return np.block([[dv_dr, dv_dv], [da_dr, da_dv]])


def RK4(x: np.ndarray, func: Callable[[np.ndarray], np.ndarray], dt: float, **kwargs) -> np.ndarray:
    """
    Computes the state at the next timestep from the current state and the continuous-time state transition function
    using Runge-Kutta 4th order integration.

    :param x: The current state vector.
    :param func: The continuous-time state transition function, \dot{x} = f_c(x).
    :param dt: The amount of time between each time step.
    :param kwargs: Additional keyword arguments to pass to the state transition function.
    :return: The state vector at the next timestep.
    """
    k1 = func(x, **kwargs)
    k2 = func(x + 0.5 * dt * k1, **kwargs)
    k3 = func(x + 0.5 * dt * k2, **kwargs)
    k4 = func(x + dt * k3, **kwargs)

    x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def RK4_jac(
    x: np.ndarray,
    func: Callable[
        [
            np.ndarray,
        ],
        np.ndarray,
    ],
    func_jac: Callable[[np.ndarray], np.ndarray],
    dt: float,
    **kwargs,
) -> np.ndarray:
    """
    Computes the Jacobian of the RK4-discretized state transition function.

    :param x: The current state vector.
    :param func: The continuous-time state transition function, \dot{x} = f_c(x).
    :param func_jac: The continuous-time state transition Jacobian function, d(f_c)/dx.
    :param dt: The amount of time between each time step.
    :param kwargs: Additional keyword arguments to pass to the state transition function.
    :return: The Jacobian of the RK4-discretized state transition function at the current state vector.
    """
    k1 = func(x, **kwargs)
    k2 = func(x + 0.5 * dt * k1, **kwargs)
    k3 = func(x + 0.5 * dt * k2, **kwargs)

    k1_jac = func_jac(x, **kwargs)
    k2_jac = func_jac(x + 0.5 * dt * k1, **kwargs) @ (np.eye(6) + 0.5 * dt * k1_jac)
    k3_jac = func_jac(x + 0.5 * dt * k2, **kwargs) @ (np.eye(6) + 0.5 * dt * k2_jac)
    k4_jac = func_jac(x + dt * k3, **kwargs) @ (np.eye(6) + dt * k3_jac)

    return np.eye(6) + (dt / 6) * (k1_jac + 2 * k2_jac + 2 * k3_jac + k4_jac)


def f(x: np.ndarray, dt: float) -> np.ndarray:
    """
    The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :param dt: The amount of time between each time step.
    :return: A numpy array of shape (6,) containing the next state (position and velocity).
    """
    return RK4(x, state_derivative, dt)


def f_jac(x: np.ndarray, dt: float) -> np.ndarray:
    """
    The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :param dt: The amount of time between each time step.
    :return: A numpy array of shape (6, 6) containing the state transition Jacobian.
    """
    return RK4_jac(x, state_derivative, state_derivative_jac, dt)


# Decorator functions
def second_order_effects(func):
    def wrapper(
        x: np.ndarray, config: dict, data_manager: ODSimulationDataManager, *args, **kwargs
    ):
        # Extract parameters from the config dictionary
        base_derivative = func(x, *args, **kwargs)

        CD = config["satellite"]["Cd"]
        AREA = config["satellite"]["area"]
        MASS = config["satellite"]["mass"]
        latest_epoch = data_manager.latest_epoch

        # Compute drag
        density = density_harris_priester(x=x, epoch=latest_epoch)
        r = x[:3]
        v = x[3:]
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            a_drag = np.zeros(3)
        else:
            a_drag = -0.5 * density * CD * AREA / (MASS * v_norm) * v

        # Compute J2
        factor = 1.5 * J2_EARTH * GM_EARTH * R_EARTH**2 / np.linalg.norm(r) ** 5
        a_J2 = np.array(
            [
                factor * r[0] * (5 * (r[2] ** 2) / r_norm**2 - 1),
                factor * r[1] * (5 * (r[2] ** 2) / r_norm**2 - 1),
                factor * r[2] * (5 * (r[2] ** 2) / r_norm**2 - 3),
            ]
        )
        updated_a = base_derivative[3:] + a_drag + a_J2
        return np.concatenate([base_derivative[:3], updated_a])

    return wrapper


def second_order_effects_jac(func):
    def wrapper(
        x: np.ndarray, config: dict, data_manager: ODSimulationDataManager, *args, **kwargs
    ):
        # Extract parameters from the config dictionary
        base_jacobian = func(x, *args, **kwargs)

        CD = config["satellite"]["Cd"]
        AREA = config["satellite"]["area"]
        MASS = config["satellite"]["mass"]
        latest_epoch = data_manager.latest_epoch

        # Compute drag
        density = density_harris_priester(x=x, epoch=latest_epoch)
        r = x[:3]
        v = x[3:]
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            a_drag = np.zeros(3)
            da_drag_dv = np.zeros((3, 3))
            da_drag_dr = np.zeros((3, 3))
        # TODO: EXTEND JACOBIAN

        # return np.block([[dv_dr, dv_dv], [da_dr_NEW, da_dv_NEW]])
        return None

    return wrapper


@second_order_effects
def f_full(x: np.ndarray, dt: float):
    return f(x, dt)


@second_order_effects_jac
def f_full_jac(x: np.ndarray, dt: float):
    return f_jac(x, dt)


"""
In your function call use:
    f_full(x, dt, config, data_manager)
    f_full_jac(x, dt, config, data_manager)
"""

"""
Functions for implementing orbital position dynamics and its jacobian under just the force of gravity.
J2 perturbations are not included.
"""

# pylint: disable=import-error
from typing import Callable

import numpy as np
from brahe.constants import GM_EARTH, J2_EARTH, R_EARTH

from dynamics.j2_dynamics import j2_derivative, j2_jacobian_auto
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from utils.earth_utils import density_harris_priester


def state_derivative(x: np.ndarray) -> np.ndarray:
    """
    The continuous-time state derivative function, dot{x} = f_c(x), for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (9,) containing the current state (position, velocity and unmodelled acceleration terms).
    :return: A numpy array of shape (9,) containing the state derivative.
    """
    r = x[:3]
    v = x[3:6]
    ua = x[6:9]

    # r_dot = v
    # v_dot = -GM_EARTH * r / np.linalg.norm(r) ** 3 
    # ua_dot = 0

    a = (-r * GM_EARTH / np.linalg.norm(r) ** 3) + ua

    return np.concatenate([v, a, np.zeros(3)])


def state_derivative_jac(x: np.ndarray) -> np.ndarray:
    """
    The continuous-time state derivative Jacobian function, d(f_c)/dx, for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :return: A numpy array of shape (6, 6) containing the state derivative Jacobian.
    """
    r = x[:3]
    r_norm = np.linalg.norm(r)

    dv_dr = np.zeros((3, 3))
    dv_dv = np.eye(3)
    dv_dua = np.zeros((3, 3))

    da_dr = (-GM_EARTH / r_norm**3) * np.eye(3) + (3 * GM_EARTH / r_norm**5) * np.outer(r, r)
    da_dv = np.zeros((3, 3))
    da_dua = np.eye(3)

    dua_dr = np.zeros((3, 3))
    dua_dv = np.zeros((3, 3))
    dua_dua = np.zeros((3, 3))

    return np.block([[dv_dr, dv_dv, dv_dua], [da_dr, da_dv, da_dua], [dua_dr, dua_dv, dua_dua]])


def RK4(x: np.ndarray, func: Callable[[np.ndarray], np.ndarray], dt: float) -> np.ndarray:
    """
    Computes the state at the next timestep from the current state and the continuous-time state transition function
    using Runge-Kutta 4th order integration.

    :param x: The current state vector.
    :param func: The continuous-time state transition function, dot{x} = f_c(x).
    :param dt: The amount of time between each time step.
    :param kwargs: Additional keyword arguments to pass to the state transition function.
    :return: The state vector at the next timestep.
    """
    k1 = func(x)
    k2 = func(x + 0.5 * dt * k1)
    k3 = func(x + 0.5 * dt * k2)
    k4 = func(x + dt * k3)

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
) -> np.ndarray:
    """
    Computes the Jacobian of the RK4-discretized state transition function.

    :param x: The current state vector.
    :param func: The continuous-time state transition function, dot{x} = f_c(x).
    :param func_jac: The continuous-time state transition Jacobian function, d(f_c)/dx.
    :param dt: The amount of time between each time step.
    :param kwargs: Additional keyword arguments to pass to the state transition function.
    :return: The Jacobian of the RK4-discretized state transition function at the current state vector.
    """
    k1 = func(x)
    k2 = func(x + 0.5 * dt * k1)
    k3 = func(x + 0.5 * dt * k2)

    k1_jac = func_jac(x)
    k2_jac = func_jac(x + 0.5 * dt * k1) @ (np.eye(9) + 0.5 * dt * k1_jac)
    k3_jac = func_jac(x + 0.5 * dt * k2) @ (np.eye(9) + 0.5 * dt * k2_jac)
    k4_jac = func_jac(x + dt * k3) @ (np.eye(9) + dt * k3_jac)

    return np.eye(9) + (dt / 6) * (k1_jac + 2 * k2_jac + 2 * k3_jac + k4_jac)


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
def second_order_effects(func: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """
    The decorator function for computing the state derivative with second order effects
    (drag and J2 perturbations).
    :param func: The state derivative function to be decorated.

    :return: A wrapper function that computes the state derivative with second order effects.
    """

    def wrapper(x: np.ndarray, config: dict, data_manager: ODSimulationDataManager) -> np.ndarray:
        """
        The wrapper function that computes the state derivative with second order effects.
        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param config: The configuration dictionary.
        :param data_manager: The ODSimulationDataManager instance.

        :return: A numpy array of shape (6,) containing the state derivative with second order effects.
        """

        base_derivative = func(x)

        # Extract parameters from the config dictionary
        # pylint: disable=invalid-name
        CD = config["satellite"]["Cd"]
        AREA = config["satellite"]["area"]
        MASS = config["satellite"]["mass"]
        latest_epoch = data_manager.latest_epoch

        # Compute drag in [kg/m^3]
        density = density_harris_priester(x=x, epoch=latest_epoch)
        r = x[:3]
        v = x[3:6]
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
        updated_a = base_derivative[3:6] + a_J2 + a_drag
        return np.concatenate([base_derivative[:3], updated_a, base_derivative[6:]])

    return wrapper


def second_order_effects_jac(func: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """
    The decorator function for computing the Jacobian of the state derivative with second order effects
    (drag and J2 perturbations).
    :param func: The state derivative function to be decorated.

    :return: A wrapper function that computes the Jacobian of the state derivative with second order effects.
    """

    def wrapper(
        x: np.ndarray,
        config: dict,
        data_manager: ODSimulationDataManager,
    ) -> np.ndarray:
        """
        The wrapper function that computes the Jacobian of the state derivative with second order effects.
        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param config: The configuration dictionary.
        :param data_manager: The ODSimulationDataManager instance.

        :return: A numpy array of shape (6, 6) containing the Jacobian of the state derivative with
        second order effects.
        """
        # Extract parameters from the config dictionary
        base_jacobian = func(x)
        # pylint: disable=invalid-name
        CD = config["satellite"]["Cd"]
        AREA = config["satellite"]["area"]
        MASS = config["satellite"]["mass"]
        latest_epoch = data_manager.latest_epoch

        # Compute drag
        density = density_harris_priester(x=x[0:6], epoch=latest_epoch)
        v = x[3:6]
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            da_drag_dv = np.zeros((3, 3))

        # Compute J2 either using autodiff or manually
        # j2da_dv = j2_derivative(x[:3])
        daj2auto_dr = j2_jacobian_auto(x[:3])

        F = -0.5 * density * CD * AREA / MASS
        da_drag_dv = F * ((np.eye(3) / v_norm) - np.outer(v, v) / v_norm**3)

        da_dr = base_jacobian[3:6, 0:3] + daj2auto_dr
        da_dv = da_drag_dv
        # da_dv = np.zeros((3, 3))

        return np.block([[base_jacobian[0:3,0:9]], [da_dr, da_dv, base_jacobian[3:6, 6:9]], [base_jacobian[6:9, 0:9]]])

    return wrapper


@second_order_effects
def state_derivative_full(x: np.ndarray) -> np.ndarray:
    """
    State derivative function with second order effects (drag and J2 perturbations).
    :param x: A numpy array of shape (6,) containing the current state (position and velocity).

    :return: A numpy array of shape (6,) containing the state derivative.
    """
    return state_derivative(x)


def f_full(
    x: np.ndarray, config: dict, data_manager: ODSimulationDataManager, dt: float
) -> np.ndarray:
    """
    The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position dynamics
    with second order effects.
    J2 perturbations and drag are included.
    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :param config: The configuration dictionary.
    :param data_manager: The ODSimulationDataManager instance.
    :param dt: The amount of time between each time step.

    :return: A numpy array of shape (6,) containing the next state (position and velocity).
    """
    dynamics = lambda state: state_derivative_full(state, config=config, data_manager=data_manager)
    return RK4(x=x, func=dynamics, dt=dt)


@second_order_effects_jac
def state_derivative_full_jac(x: np.ndarray) -> np.ndarray:
    """
    State derivative Jacobian function with second order effects (drag and J2 perturbations).
    :param x: A numpy array of shape (6,) containing the current state (position and velocity).

    :return: A numpy array of shape (6, 6) containing the state derivative Jacobian.
    """
    return state_derivative_jac(x)


def f_full_jac(
    x: np.ndarray, config: dict, data_manager: ODSimulationDataManager, dt: float
) -> np.ndarray:
    """
    The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position dynamics
    with second order effects.
    J2 perturbations and drag are included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :param config: The configuration dictionary.
    :param data_manager: The ODSimulationDataManager instance.
    :param dt: The amount of time between each time step.

    :return: A numpy array of shape (6, 6) containing the state transition Jacobian.
    """
    dynamics = lambda state: state_derivative_full(state, config=config, data_manager=data_manager)
    jacobian = lambda state: state_derivative_full_jac(
        state, config=config, data_manager=data_manager
    )
    return RK4_jac(x=x, func=dynamics, func_jac=jacobian, dt=dt)

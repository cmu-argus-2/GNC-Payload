from brahe.constants import GM_EARTH
import numpy as np
import quaternion

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.math_utils import dqdot_dq_deriv, left_q


"""
Functions for implementing orbital position and attitude dynamics and its jacobian under just the force of gravity.
J2 perturbations are not included.
"""


def quat_derivative(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the time derivative of quaternion q given angular velocity w.
    q = [q0, q1, q2, q3]
    w = [wx, wy, wz]
    """
    dq = 0.5 * left_q(q) @ np.append(np.zeros((1,3)), np.eye(3), axis=0) @ w
    return dq


def state_derivative(x: np.ndarray) -> np.ndarray:
    """
    The continuous-time state derivative function, \dot{x} = f_c(x), for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :return: A numpy array of shape (6,) containing the state derivative.
    """
    r = x[:3]
    v = x[3:6]
    a = -r * GM_EARTH / np.linalg.norm(r) ** 3

    q = x[6:10]
    w = x[10:13]

    dq = quat_derivative(q, w)
    dw = np.zeros(3)  
    # For now (?) we are not including angular velocity dynamics assuming constant angular velocity.

    return np.concatenate([v, a, dq, dw])


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
    dv_dq = np.zeros((3, 3))

    da_dr = (-GM_EARTH / r_norm**3) * np.eye(3) + (3 * GM_EARTH / r_norm**5) * np.outer(r, r)
    da_dv = np.zeros((3, 3))
    da_dq = np.zeros((3, 3))

    dqdot_dr = np.zeros((3, 3))
    dqdot_dv = np.zeros((3, 3))
    dqdot_dq = np.zeros((3, 3))
    # dqdot_dw = 0.5 * left_q(q) @ np.append(np.zeros((1,3)), np.eye(3), axis=0)
    # dqdot_dw = np.zeros((3, 3))

    # dwdot_dr = np.zeros((3, 3))
    # dwdot_dv = np.zeros((3, 3))
    # dwdot_dq = np.zeros((3, 3))
    # dwdot_dw = np.zeros((3, 3))

    return np.block(
        [
            [dv_dr, dv_dv, dv_dq],
            [da_dr, da_dv, da_dq],
            [dqdot_dr, dqdot_dv, dqdot_dq],
        ]
    )


def RK4(x, func, dt):
    """
    Computes the state at the next timestep from the current state and the continuous-time state transition function
    using Runge-Kutta 4th order integration.

    :param x: The current state vector.
    :param func: The continuous-time state transition function, \dot{x} = f_c(x).
    :param dt: The amount of time between each time step.
    :return: The state vector at the next timestep.
    """
    k1 = func(x)
    k2 = func(x + 0.5 * dt * k1)
    k3 = func(x + 0.5 * dt * k2)
    k4 = func(x + dt * k3)

    x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def RK4_jac(x, func, func_jac, dt):
    """
    Computes the Jacobian of the RK4-discretized state transition function.

    :param x: The current state vector.
    :param func: The continuous-time state transition function, \dot{x} = f_c(x).
    :param func_jac: The continuous-time state transition Jacobian function, d(f_c)/dx.
    :param dt: The amount of time between each time step.
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
    The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position and attitude dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (13,) containing the current state in form [position, velocity, quaternion, angular_velocity].
    :param dt: The amount of time between each time step.
    :return: A numpy array of shape (13,) containing the next state (position and velocity).
    """
    return RK4(x, state_derivative, dt)


def f_jac(x: np.ndarray, dt: float) -> np.ndarray:
    """
    The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position and attitude dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (13,) containing the current state in form [position, velocity, quaternion, angular_velocity].
    :param dt: The amount of time between each time step.
    :return: A numpy array of shape (13, 13) containing the state transition Jacobian.
    """
    return RK4_jac(x, state_derivative, state_derivative_jac, dt)

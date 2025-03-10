"""
Functions for implementing orbital position dynamics and its jacobian under just the force of gravity.
J2 perturbations are not included.
"""

# pylint: disable=import-error
from functools import partial
from typing import Callable

import numpy as np
from brahe import Epoch
from brahe.constants import GM_EARTH

from dynamics.drag_dynamics import drag_dynamics, drag_jacobian
from dynamics.j2_dynamics import j2_dynamics, j2_jacobian_auto, j2_jacobian_manual

# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments


class Dynamics:
    """
    This class contains the orbital dynamics functions and second order perturbations. Basic orbital dynamics are
    implemented as static methods so that they can be used without instantiating the class.
    """

    def __init__(
        self,
        config: dict,
        use_drag: bool,
        use_j2: bool,
    ) -> None:
        """
        Initialize the Dynamics class.

        :param config: The configuration dictionary.
        :param use_drag: Whether to use drag in the dynamics.
        :param use_j2: Whether to use J2 perturbations in the dynamics.
        :return: None
        """
        self.use_drag = use_drag
        self.use_j2 = use_j2
        self.drag_const = (
            -0.5
            * config["satellite"]["Cd"]
            * config["satellite"]["area"]
            / config["satellite"]["mass"]
        )

        # If no measurement was made in the previous measurement step, set the unmodelled accelerations to zero
        self.nominal_density = 1e-12

    @staticmethod
    def state_derivative(x: np.ndarray) -> np.ndarray:
        """
        The continuous-time state derivative function, dot{x} = f_c(x), for orbital position dynamics under gravity.
        No perturbations are included.

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :return: A numpy array of shape (6,) containing the state derivative.
        """
        r = x[0:3]
        v = x[3:6]

        a = -r * GM_EARTH / np.linalg.norm(r) ** 3

        return np.concatenate([v, a])

    @staticmethod
    def state_derivative_jac(x: np.ndarray) -> np.ndarray:
        """
        The continuous-time state derivative Jacobian function, d(f_c)/dx, for orbital position dynamics under gravity.
        No perturbations are included.

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :return: A numpy array of shape (6, 6) containing the state derivative Jacobian.
        """
        r = x[0:3]
        r_norm = np.linalg.norm(r)
        dv_dr = np.zeros((3, 3))
        da_dr = (-GM_EARTH / r_norm**3) * np.eye(3) + (3 * GM_EARTH / r_norm**5) * np.outer(r, r)
        dv_dv = np.eye(3)
        da_dv = np.zeros((3, 3))
        return np.block([[dv_dr, dv_dv], [da_dr, da_dv]])

    @staticmethod
    def RK4(x: np.ndarray, func: Callable[[np.ndarray], np.ndarray], dt: float) -> np.ndarray:
        """
        Computes the state at the next timestep from the current state and the continuous-time state transition function
        using Runge-Kutta 4th order integration.

        :param x: The current state vector.
        :param func: The continuous-time state transition function, dot{x} = f_c(x).
        :param dt: The amount of time between each time step.
        :return: The state vector at the next timestep.
        """
        k1 = func(x)
        k2 = func(x + 0.5 * dt * k1)
        k3 = func(x + 0.5 * dt * k2)
        k4 = func(x + dt * k3)

        x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    @staticmethod
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
        :return: The Jacobian of the RK4-discretized state transition function at the current state vector.
        """
        k1 = func(x)
        k2 = func(x + 0.5 * dt * k1)
        k3 = func(x + 0.5 * dt * k2)

        k1_jac = func_jac(x)
        k2_jac = func_jac(x + 0.5 * dt * k1) @ (np.eye(x.shape[0]) + 0.5 * dt * k1_jac)
        k3_jac = func_jac(x + 0.5 * dt * k2) @ (np.eye(x.shape[0]) + 0.5 * dt * k2_jac)
        k4_jac = func_jac(x + dt * k3) @ (np.eye(x.shape[0]) + dt * k3_jac)

        return np.eye(x.shape[0]) + (dt / 6) * (k1_jac + 2 * k2_jac + 2 * k3_jac + k4_jac)

    @staticmethod
    def f(x: np.ndarray, dt: float) -> np.ndarray:
        """
        The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position dynamics under gravity.
        No perturbations are included.

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param dt: The amount of time between each time step.
        :return: A numpy array of shape (6,) containing the next state (position and velocity).
        """
        return Dynamics.RK4(x, Dynamics.state_derivative, dt)

    @staticmethod
    def f_jac(x: np.ndarray, dt: float) -> np.ndarray:
        """
        The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position dynamics under gravity.
        No perturbations are included.

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param dt: The amount of time between each time step.
        :return: A numpy array of shape (6, 6) containing the state transition Jacobian.
        """
        return Dynamics.RK4_jac(x, Dynamics.state_derivative, Dynamics.state_derivative_jac, dt)

    def perturbed_state_derivative(self, x: np.ndarray, epoch: Epoch = None) -> np.ndarray:
        """
        The continuous-time state derivative function, dot{x} = f_c(x), for orbital position dynamics under gravity
        and the configured perturbations.

        :param x: A numpy array of shape (6,) containing the current state (position, velocity).
        :param epoch: The current time epoch. Can be None if the configured perturbations do not require it.

        :return: A numpy array of shape (6,) containing the full state derivative.
        """
        base_derivative = Dynamics.state_derivative(x)
        r = x[0:3]
        v = x[3:6]
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)

        updated_a = base_derivative[3:6]

        # Compute drag
        if self.use_drag and np.isclose(v_norm, 0):
            if epoch is None:
                raise ValueError("Epoch is required to compute drag")
            a_drag_gt = drag_dynamics(x=x, drag_const=self.drag_const, latest_epoch=epoch)

            updated_a += a_drag_gt

        # Compute J2
        if self.use_j2 and np.isclose(r_norm, 0):
            a_J2_gt = j2_dynamics(r)

            updated_a += a_J2_gt

        return np.concatenate([base_derivative[0:3], updated_a])

    def perturbed_state_derivative_jac(self, x: np.ndarray, epoch: Epoch = None) -> np.ndarray:
        """
        The continuous-time state derivative Jacobian function, d(f_c)/dx, for orbital position dynamics under gravity
        and the configured perturbations.

        :param x: A numpy array of shape (6,) containing the current state (position, velocity).
        :param epoch: The current time epoch. Can be None if the configured perturbations do not require it.

        :return: A numpy array of shape (6,6) containing the state derivative Jacobian.
        """
        base_jacobian = Dynamics.state_derivative_jac(x)

        v = x[3:6]
        v_norm = np.linalg.norm(v)

        da_dr = base_jacobian[3:6, 0:3]
        da_dv = base_jacobian[3:6, 3:6]

        # Compute drag
        if self.use_drag and np.isclose(v_norm, 0):
            if epoch is None:
                raise ValueError("Epoch is required to compute drag jacobian")
            da_drag_gt_dv = drag_jacobian(x=x, drag_const=self.drag_const, latest_epoch=epoch)

            da_dv += da_drag_gt_dv

        # Compute J2 either using autodiff or manually
        if self.use_j2:
            # da_J2_gt_dr = j2_derivative_manual(x[:3])
            da_J2_gt_dr = j2_jacobian_auto(x[0:3])

            da_dr += da_J2_gt_dr

        return np.block([[base_jacobian[0:3, 0:6]], [da_dr, da_dv]])

    def perturbed_f(self, x: np.ndarray, dt: float, epoch: Epoch = None) -> np.ndarray:
        """
        The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position dynamics under gravity
        and the configured perturbations.

        :param x: A numpy array of shape (6,) containing the current state (position, velocity).
        :param dt: The amount of time between each time step.
        :param epoch: The current time epoch. Can be None if the configured perturbations do not require it.

        :return: A numpy array of shape (6,) containing the next state (position, velocity).
        """
        func = (
            partial(self.perturbed_state_derivative, epoch=epoch)
            if self.use_drag
            else self.perturbed_state_derivative
        )
        return Dynamics.RK4(x=x, func=func, dt=dt)

    def perturbed_f_jac(self, x: np.ndarray, dt: float, epoch: Epoch = None) -> np.ndarray:
        """
        The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position dynamics under gravity
        and the configured perturbations.

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param dt: The amount of time between each time step.
        :param epoch: The current time epoch.Can be None if the configured perturbations do not require it.

        :return: A numpy array of shape (6, 6) containing the state transition Jacobian.
        """

        func = (
            partial(self.perturbed_state_derivative, epoch=epoch)
            if self.use_drag
            else self.perturbed_state_derivative
        )
        func_jac = (
            partial(self.perturbed_state_derivative_jac, epoch=epoch)
            if self.use_drag
            else self.perturbed_state_derivative_jac
        )
        return Dynamics.RK4_jac(
            x=x,
            func=func,
            func_jac=func_jac,
            dt=dt,
        )

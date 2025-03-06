"""
Functions for implementing orbital position dynamics and its jacobian under just the force of gravity.
J2 perturbations are not included.
"""

# pylint: disable=import-error
from typing import Callable

import numpy as np
from brahe.constants import GM_EARTH
from brahe.epoch import Epoch

from dynamics.drag_dynamics import drag_dynamics, drag_jacobian
from dynamics.j2_dynamics import j2_dynamics, j2_jacobian_auto, j2_jacobian_manual
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager


class Dynamics:
    """
    This class contains the orbital dynamics functions. Basic orbital dynamics are
    implemented as static methods so that they can be used without instantiating the class.
    """

    def __init__(
        self,
        config: dict,
        data_manager: ODSimulationDataManager,
        use_unmodelled_a: bool,
        use_drag: bool,
        use_j2: bool,
    ) -> None:
        """
        Initialize the OrbitalDynamics class.

        :param config: The configuration dictionary.
        :param data_manager: The ODSimulationDataManager instance.
        :param use_unmodelled_a: Whether to use unmodelled accelerations in the dynamics.
        :param use_drag: Whether to use drag in the dynamics.
        :param use_j2: Whether to use J2 perturbations in the dynamics.
        :return: None
        """
        self.data_manager = data_manager
        self.use_unmodelled_a = use_unmodelled_a
        self.use_drag = use_drag
        self.use_j2 = use_j2
        self.drag_const = (
            -0.5
            * config["satellite"]["Cd"]
            * config["satellite"]["area"]
            / config["satellite"]["mass"]
        )

        # If no measurement was made in the previous measurement step, set the unmodelled accelerations to zero
        self.no_previous_measurement = False
        self.nominal_density = 1e-12

        if use_unmodelled_a:
            self.ua_std_dev = 1e-5

    @staticmethod
    def state_derivative(x: np.ndarray) -> np.ndarray:
        """
        The continuous-time state derivative function, dot{x} = f_c(x), for orbital position dynamics under gravity.
        J2 perturbations are not included.

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
        J2 perturbations are not included.

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
        J2 perturbations are not included.

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param dt: The amount of time between each time step.
        :return: A numpy array of shape (6,) containing the next state (position and velocity).
        """
        return Dynamics.RK4(x, Dynamics.state_derivative, dt)

    @staticmethod
    def f_jac(x: np.ndarray, dt: float) -> np.ndarray:
        """
        The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position dynamics under gravity.
        J2 perturbations are not included.

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param dt: The amount of time between each time step.
        :return: A numpy array of shape (6, 6) containing the state transition Jacobian.
        """
        return Dynamics.RK4_jac(x, Dynamics.state_derivative, Dynamics.state_derivative_jac, dt)

    def perturbed_state_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        The continuous-time state derivative function, dot{x} = f_c(x), for orbital position dynamics under gravity,
        J2 perturbations and gravity.

        :param x: A numpy array of shape (6,) or (9,) containing the current state position, velocity, (unmodelled_accelerations).
        :return: A numpy array of shape (6,) or (9,) containing the full state derivative.
        """
        base_derivative = Dynamics.state_derivative(x)
        r = x[0:3]
        v = x[3:6]
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)

        updated_a = base_derivative[3:6]

        # Compute drag
        if self.use_drag and np.isclose(v_norm, 0):
            a_drag_gt = drag_dynamics(
                x=x, drag_const=self.drag_const, latest_epoch=self.data_manager.latest_epoch
            )

            updated_a += a_drag_gt

        # Compute J2
        if self.use_j2 and np.isclose(r_norm, 0):
            a_J2_gt = j2_dynamics(r)

            updated_a += a_J2_gt

        # Compute unmodelled accelerations
        if self.use_unmodelled_a:
            unmodelled_a = x[6:9]
            ua_dot = np.random.normal(0, self.ua_std_dev, 3)

            updated_a += unmodelled_a

            return np.concatenate([base_derivative[0:3], updated_a, ua_dot])

        return np.concatenate([base_derivative[0:3], updated_a])

    def perturbed_state_derivative_jac(self, x: np.ndarray) -> np.ndarray:
        """
        The continuous-time state derivative Jacobian function, d(f_c)/dx, for orbital position dynamics under gravity,
        J2 perturbations and gravity.

        :param x: A numpy array of shape (6,) or (9,) containing the current state position, velocity, (unmodelled_accelerations).
        :return: A numpy array of shape (6,6) or (9,9) containing the state derivative Jacobian.
        """
        base_jacobian = Dynamics.state_derivative_jac(x)

        v = x[3:6]
        v_norm = np.linalg.norm(v)

        da_dr = base_jacobian[3:6, 0:3]
        da_dv = base_jacobian[3:6, 3:6]

        # Compute drag
        if self.use_drag and np.isclose(v_norm, 0):
            da_drag_gt_dv = drag_jacobian(
                x=x, drag_const=self.drag_const, latest_epoch=self.data_manager.latest_epoch
            )

            da_dv += da_drag_gt_dv

        # Compute J2 either using autodiff or manually
        if self.use_j2:
            # da_J2_gt_dr = j2_derivative_manual(x[:3])
            da_J2_gt_dr = j2_jacobian_auto(x[0:3])

            da_dr += da_J2_gt_dr

        # Compute unmodelled accelerations
        if self.use_unmodelled_a:
            dv_dua = np.zeros((3, 3))
            da_dua = np.eye(3)

            dua_dr = np.zeros((3, 3))
            dua_dv = np.zeros((3, 3))
            dua_dua = np.zeros((3, 3))

            return np.block(
                [
                    [base_jacobian[0:3, 0:6], dv_dua],
                    [da_dr, da_dv, da_dua],
                    [dua_dr, dua_dv, dua_dua],
                ]
            )

        return np.block([[base_jacobian[0:3, 0:6]], [da_dr, da_dv]])

        # TODO: incorporate drag estimate into the jacobian
        # dest_drag = np.zeros((3,9))
        # dv_dest_drag = np.zeros((3,1))

        # da_dest_drag = self.drag_const * self.nominal_density * v / v_norm
        # dua_dest_drag = np.zeros((3,1))
        # dest_drag_dest_drag = np.eye((1))

    def perturbed_f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position dynamics
        with second order effects.
        J2 perturbations and drag can be included.
        :param x: A numpy array of shape (9,) containing the current state (position, velocity and unmodelled accelerations).
        :param dt: The amount of time between each time step.

        :return: A numpy array of shape (9,) containing the next state (position, velocity and unmodelled accelerations).
        """
        return Dynamics.RK4(x=x, func=self.perturbed_state_derivative, dt=dt)

    def perturbed_f_jac(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position dynamics
        with second order effects.
        J2 perturbations and drag can be included.

        :param x: A numpy array of shape (9,) containing the current state (position and velocity).
        :param dt: The amount of time between each time step.

        :return: A numpy array of shape (9, 9) containing the state transition Jacobian.
        """
        return Dynamics.RK4_jac(
            x=x,
            func=self.perturbed_state_derivative,
            func_jac=self.perturbed_state_derivative_jac,
            dt=dt,
        )

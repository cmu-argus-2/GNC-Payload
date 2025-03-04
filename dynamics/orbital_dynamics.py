"""
Functions for implementing orbital position dynamics and its jacobian under just the force of gravity.
J2 perturbations are not included.
"""

# pylint: disable=import-error
from typing import Callable

import numpy as np
from brahe import Epoch
from brahe.constants import GM_EARTH, J2_EARTH, R_EARTH

from dynamics.j2_dynamics import j2_derivative, j2_jacobian_auto
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from utils.earth_utils import density_harris_priester


class OrbitalDynamics:
    """
    This class contains the orbital dynamics functions. Basic orbital dynamics are
    implemented as static methods so that they can be used without instantiating the class.
    """

    def __init__(
        self,
        config: dict,
        data_manager: ODSimulationDataManager,
        use_drag: bool,
        use_j2: bool,
    ) -> None:
        """
        Initialize the OrbitalDynamics class.

        :param config: The configuration dictionary.
        :param data_manager: The ODSimulationDataManager instance.
        :param use_drag: Whether to use drag in the dynamics.
        :param use_j2: Whether to use J2 perturbations in the dynamics.
        :return: None
        """
        self.mass = config["satellite"]["mass"]
        self.area = config["satellite"]["area"]
        self.drag_coefficient = config["satellite"]["Cd"]
        self.data_manager = data_manager
        self.use_drag = use_drag
        self.use_j2 = use_j2

        # If no measurement was made in the previous measurement step, set the unmodelled accelerations to zero
        self.no_previous_measurement = False
        self.nominal_density = 1e-12

        self.j2_factor = 1.5 * J2_EARTH * GM_EARTH * R_EARTH**2

    @staticmethod
    def state_derivative(x: np.ndarray) -> np.ndarray:
        """
        The continuous-time state derivative function, dot{x} = f_c(x), for orbital position dynamics under gravity.
        J2 perturbations are not included.

        :param x: A numpy array of shape (6,) containing the current state (position, velocity and unmodelled acceleration terms).
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
        dv_dv = np.eye(3)

        da_dr = (-GM_EARTH / r_norm**3) * np.eye(3) + (3 * GM_EARTH / r_norm**5) * np.outer(r, r)
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
        :param kwargs: Additional keyword arguments to pass to the state transition function.
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
        return OrbitalDynamics.RK4(x=x, func=OrbitalDynamics.state_derivative, dt=dt)

    @staticmethod
    def f_jac(x: np.ndarray, dt: float) -> np.ndarray:
        """
        The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position dynamics under gravity.
        J2 perturbations are not included.

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param dt: The amount of time between each time step.
        :return: A numpy array of shape (6, 6) containing the state transition Jacobian.
        """
        return OrbitalDynamics.RK4_jac(
            x=x,
            func=OrbitalDynamics.state_derivative,
            func_jac=OrbitalDynamics.state_derivative_jac,
            dt=dt,
        )

    def full_state_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        The continuous-time state derivative function, dot{x} = f_c(x), for orbital position dynamics under gravity,
        J2 perturbations and gravity.

        :param x: A numpy array of shape (9,) containing the current state (position, velocity, unmodelled_accelerations).
        :return: A numpy array of shape (9,) containing the full state derivative.
        """
        base_derivative = OrbitalDynamics.state_derivative(x)
        # TODO: Replace these with the correct files when they ready to rebase
        # Then use the lambda trick to create a function that takes in the config and latest_epoch and can use the same RK4
        # and RK4_jac functions# Compute drag in [kg/m^3]
        r = x[0:3]
        v = x[3:6]
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)

        a_drag = np.zeros(3)
        a_J2 = np.zeros(3)

        # Compute drag
        if self.use_drag and v_norm != 0:
            density = density_harris_priester(x=x, epoch=self.data_manager.latest_epoch)
            a_drag = -0.5 * density * self.drag_coefficient * self.area / (self.mass * v_norm) * v

        # Compute J2
        if self.use_j2:
            factor = self.j2_factor / np.linalg.norm(r) ** 5
            a_J2 = np.array(
                [
                    factor * r[0] * (5 * (r[2] ** 2) / r_norm**2 - 1),
                    factor * r[1] * (5 * (r[2] ** 2) / r_norm**2 - 1),
                    factor * r[2] * (5 * (r[2] ** 2) / r_norm**2 - 3),
                ]
            )

        # Compute unmodelled accelerations
        unmodelled_a = x[6:9]
        ua_dot = np.random.normal(0, 1e-5, 3)

        updated_a = base_derivative[3:6] + a_J2 + a_drag + unmodelled_a

        return np.concatenate([base_derivative[0:3], updated_a, ua_dot])

    def full_state_derivative_jac(self, x: np.ndarray) -> np.ndarray:
        """
        The continuous-time state derivative Jacobian function, d(f_c)/dx, for orbital position dynamics under gravity,
        J2 perturbations and gravity.

        :param x: A numpy array of shape (9,) containing the current state (position, velocity, unmodelled_accelerations).
        :return: A numpy array of shape (9, 9) containing the state derivative Jacobian.
        """
        base_jacobian = OrbitalDynamics.state_derivative_jac(x)

        v = x[3:6]
        v_norm = np.linalg.norm(v)

        da_drag_dv = np.zeros((3, 3))
        da_j2_dr = np.zeros((3, 3))

        # Compute drag
        if self.use_drag and v_norm != 0:
            density = density_harris_priester(x=x[0:6], epoch=self.data_manager.latest_epoch)
            F = -0.5 * density * self.drag_coefficient * self.area / self.mass
            da_drag_dv = F * ((np.eye(3) / v_norm) - np.outer(v, v) / v_norm**3)

        # Compute J2 either using autodiff or manually
        if self.use_j2:
            # da_j2_dr = j2_derivative(x[:3])
            da_j2_dr = j2_jacobian_auto(x[:3])

        da_dr = base_jacobian[3:6, 0:3] + da_j2_dr
        da_dv = base_jacobian[3:6, 3:6] + da_drag_dv

        dv_dua = np.zeros((3, 3))
        da_dua = np.eye(3)

        dua_dr = np.zeros((3, 3))
        dua_dv = np.zeros((3, 3))
        dua_dua = np.zeros((3, 3))

        # dest_drag = np.zeros((3,9))
        # dv_dest_drag = np.zeros((3,1))
        # da_dest_drag = -0.5 * self.nominal_density * self.drag_coefficient * self.area / (self.mass * v_norm) * v
        # dua_dest_drag = np.zeros((3,1))
        # dest_drag_dest_drag = np.eye((1))

        # TODO: incorporate drag estimate into the jacobian

        return np.block(
            [
                [base_jacobian[0:3, 0:6], dv_dua],
                [da_dr, da_dv, da_dua],
                [dua_dr, dua_dv, dua_dua],
            ]
        )

    def full_f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position dynamics
        with second order effects.
        J2 perturbations and drag can be included.
        :param x: A numpy array of shape (9,) containing the current state (position, velocity and unmodelled accelerations).
        :param dt: The amount of time between each time step.

        :return: A numpy array of shape (9,) containing the next state (position, velocity and unmodelled accelerations).
        """
        dynamics = lambda state: self.full_state_derivative(state)
        return OrbitalDynamics.RK4(x=x, func=dynamics, dt=dt)

    def full_f_jac(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position dynamics
        with second order effects.
        J2 perturbations and drag can be included.

        :param x: A numpy array of shape (9,) containing the current state (position and velocity).
        :param dt: The amount of time between each time step.

        :return: A numpy array of shape (9, 9) containing the state transition Jacobian.
        """
        dynamics = lambda state: self.full_state_derivative(state)
        jacobian = lambda state: self.full_state_derivative_jac(state)
        return OrbitalDynamics.RK4_jac(x=x, func=dynamics, func_jac=jacobian, dt=dt)

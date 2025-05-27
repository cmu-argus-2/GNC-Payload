"""
This module contains the orbital dynamics to a higher fidelity than the standard, using the brahe library.
Includes higher order spherical harmonics, third body accelerations, and solar radiation pressure.
Drag still needs to be treated separately.
"""

from functools import partial
from typing import Callable

import brahe
import numpy as np
import quaternion
from brahe.constants import GM_EARTH
from brahe.epoch import Epoch
from brahe.orbit_dynamics import gravity, srp

from dynamics.srp_dynamics import srp_acceleration
from dynamics.drag_dynamics import drag_dynamics
from dynamics.third_body_dynamics import moon_gravity, sun_gravity
from utils.math_utils import G

GM_EARTH = GM_EARTH / 1e9  # Convert to km^3/s^2


class BraheDynamics:
    """
    This class contains the orbital dynamics functions implemented with Brahe.
    """

    def __init__(
        self,
        config: dict,
        use_drag: bool,
        use_sun_grav: bool,
        use_moon_grav: bool,
        use_srp: bool,
    ) -> None:
        """
        Initialize the Dynamics class.

        :param config: The configuration dictionary.
        :param use_drag: Whether to use drag in the dynamics.
        :param use_sun_grav: Whether to use the sun's gravity in the dynamics.
        :param use_moon_grav: Whether to use the moon's gravity in the dynamics.
        :param use_srp: Whether to use solar radiation pressure in the dynamics.
        :return: None
        """
        self.use_drag = use_drag
        self.use_sun_grav = use_sun_grav
        self.use_moon_grav = use_moon_grav
        self.use_srp = use_srp
        self.drag_const = (
            -0.5
            * config["satellite"]["Cd"]
            * config["satellite"]["area"]
            / config["satellite"]["mass"]
        )
        self.area = config["satellite"]["area"]
        self.mass = config["satellite"]["mass"]

    @property
    def require_epoch(self) -> bool:
        """
        :return: True if the configured perturbations require the current time epoch, False otherwise.
        """
        return self.use_drag or self.use_sun_grav or self.use_moon_grav or self.use_srp

    def RK4(self, x: np.ndarray, func: Callable[[np.ndarray], np.ndarray], dt: float) -> np.ndarray:
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

        # Ensure the quaternion is normalized
        x_next[6:10] = x_next[6:10] / np.linalg.norm(x_next[6:10])

        return x_next

    def perturbed_state_derivative(
        self, x: np.ndarray, w: np.ndarray, epoch: Epoch = None
    ) -> np.ndarray:
        """
        The continuous-time state derivative function, dot{x} = f_c(x), for orbital position dynamics under gravity
        and the configured perturbations.

        :param x: A numpy array of shape (10,) containing the current state (position, velocity, quaternion).
        :param w: A numpy array of shape (3,) containing the angular velocity.
        :param epoch: The current time epoch. Can be None if the configured perturbations do not require it.

        :return: A numpy array of shape (10,) containing the full state derivative.
        """
        # base_derivative = Dynamics.state_derivative(x)
        r = x[0:3] * 1e3  # Convert to meters
        v = x[3:6]
        v_norm = np.linalg.norm(v)
        q = x[6:10]
        rot_matrix = quaternion.as_rotation_matrix(quaternion.as_quat_array(q))

        updated_a = (
            gravity.accel_gravity(x=r, R_i2b=rot_matrix, n_max=5, m_max=5) / 1e3
        )  # Convert to km/s^2

        # Compute drag
        if self.use_drag and not np.isclose(v_norm, 0):
            if epoch is None:
                raise ValueError("Epoch is required to compute drag")
            a_drag_gt = drag_dynamics(x=x[0:6], drag_const=self.drag_const, latest_epoch=epoch)

            updated_a += a_drag_gt

        # Compute third body gravity
        if self.use_sun_grav:
            if epoch is None:
                raise ValueError("Epoch is required to compute sun gravitational effects")
            a_sun_gt = sun_gravity(r_sat=x[0:3], epoch=epoch)

            updated_a += a_sun_gt

        if self.use_moon_grav:
            if epoch is None:
                raise ValueError("Epoch is required to compute moon gravitational effects")
            a_moon_gt = moon_gravity(r_sat=x[0:3], epoch=epoch)

            updated_a += a_moon_gt

        # Compute solar radiation pressure
        if self.use_srp:
            if epoch is None:
                raise ValueError("Epoch is required to compute solar radiation pressure")
            a_srp_gt = srp_acceleration(
                r_sat=x[0:3], area=self.area, mass=self.mass, epoch=epoch
            )

            updated_a += a_srp_gt

        q_dot = 0.5 * G(q) @ w

        return np.concatenate([v, updated_a, q_dot])

    def perturbed_f(
        self, x: np.ndarray, dt: float, w: np.ndarray, epoch: Epoch = None
    ) -> np.ndarray:
        """
        The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position dynamics under gravity
        and the configured perturbations.

        :param x: A numpy array of shape (10,) containing the current state (position, velocity, quaternion).
        :param dt: The amount of time between each time step.
        :param w: A numpy array of shape (3,) containing the angular velocity.
        :param epoch: The current time epoch. Can be None if the configured perturbations do not require it.

        :return: A numpy array of shape (10,) containing the next state (position, velocity, quaternion).
        """
        func = (
            partial(self.perturbed_state_derivative, w=w, epoch=epoch)
            if self.require_epoch
            else self.perturbed_state_derivative
        )
        return self.RK4(x=x, func=func, dt=dt)

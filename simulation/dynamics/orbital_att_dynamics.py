"""
Functions for implementing orbital position dynamics and its jacobian under just the force of gravity.
J2 perturbations are not included.
"""

# pylint: disable=import-error
from typing import Callable

import numpy as np
from brahe import R_EARTH, Epoch
from brahe.constants import GM_EARTH
from dynamics.drag_dynamics import drag_dynamics, drag_jacobian
from dynamics.grav_potential_dynamics import (
    j2_dynamics,
    j2_jacobian_manual,
    j3_dynamics,
    j3_jacobian_auto,
    j4_dynamics,
    j4_jacobian_auto,
)
from dynamics.third_body_dynamics import (
    moon_gravity,
    moon_gravity_jac,
    sun_gravity,
    sun_gravity_jac,
)
from utils.earth_utils import density_harris_priester
from utils.math_utils import left_q_3, right_q, skew

# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
GM_EARTH = GM_EARTH / 1e9  # Convert to km^3/s^2
REF_HEIGHT = 600  # km
NOMINAL_DENSITY = 1e-5  # kg/m^3
R_EARTH = R_EARTH / 1e3  # km


class DynamicsIDX:
    NX = 13
    POS = slice(0, 3)
    RX = 0
    RY = 1
    RZ = 2
    VEL = slice(3, 6)
    VX = 3
    VY = 4
    VZ = 5
    ORB = slice(0, 6)
    QUAT = slice(6, 10)
    QW = 6
    QX = 7
    QY = 8
    QZ = 9
    OMEGA = slice(10, 13)
    OMEGA_X = 10
    OMEGA_Y = 11
    OMEGA_Z = 12
    ROT = slice(6, 13)
    GYR_BIAS = None
    GYR_BIAS_X = None
    GYR_BIAS_Y = None
    GYR_BIAS_Z = None

    def __init__(self, has_gyro_bias: bool = False) -> None:
        if has_gyro_bias:
            self.GYR_BIAS = slice(13, 16)
            self.GYR_BIAS_X = 13
            self.GYR_BIAS_Y = 14
            self.GYR_BIAS_Z = 15
            self.NX = 16


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
        use_j34: bool,
        use_sun_grav: bool,
        use_moon_grav: bool,
        include_gyro_bias: bool = False,
        gyro_bias_tau: float = np.inf,
        gyro_bias_std: float = 0.0,
    ) -> None:
        """
        Initialize the Dynamics class.

        :param config: The configuration dictionary.
        :param use_drag: Whether to use drag in the dynamics.
        :param use_j2: Whether to use J2 perturbations in the dynamics.
        :param use_j34: Whether to use J3 and J4 perturbations in the dynamics.
        :param use_sun_grav: Whether to use the sun's gravity in the dynamics.
        :param use_moon_grav: Whether to use the moon's gravity in the dynamics.
        :return: None
        """
        self.use_drag = use_drag
        self.use_j2 = use_j2
        self.use_j34 = use_j34
        self.use_sun_grav = use_sun_grav
        self.use_moon_grav = use_moon_grav
        self.drag_const = (
            -0.5
            * config["satellite"]["Cd"]
            * config["satellite"]["area"]
            / config["satellite"]["mass"]
        )
        self.I_sat = np.array(config["satellite"]["inertia"])
        self.I_sat_inv = np.linalg.inv(self.I_sat)
        self.has_gyro_bias = include_gyro_bias
        self.gyro_bias_tau = config["satellite"].get("gyro_bias_tau", 0.0)
        self.gyro_bias_std = config["satellite"].get("gyro_bias_std", 0.0)
        # self.CoPM = np.array(config["satellite"].get("CoPM", [0, 0, 0]))
        # self.num_MTBs = config["satellite"].get("num_MTBs", 0)
        # Add other satellite parameters as needed

    @property
    def require_epoch(self) -> bool:
        """
        :return: True if the configured perturbations require the current time epoch, False otherwise.
        """
        return self.use_drag or self.use_sun_grav or self.use_moon_grav

    def state_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        The continuous-time state derivative function, dot{x} = f_c(x), for orbital position and attitude dynamics.
        :param x: A numpy array of shape (13,) containing the current state
        (position, velocity, quaternion, angular velocity).
        :return: A numpy array of shape (13,) containing the state derivative.
        """
        r = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]

        # Translational acceleration
        a = -r * GM_EARTH / np.linalg.norm(r) ** 3

        # Quaternion derivative
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        qdot = 0.5 * left_q_3(q) @ w

        # Torques
        tau = np.zeros(3)

        # Angular acceleration
        h_sc = self.I_sat @ w
        omegadot = self.I_sat_inv @ (tau - np.cross(w, h_sc))

        x_dot = np.concatenate([v, a, qdot, omegadot])  # pylint: disable=W0101

        if self.has_gyro_bias:
            gyro_bias_dot = (
                -x[13:16] + np.random.normal(0, self.gyro_bias_std, (3,))
            ) / self.gyro_bias_tau
            x_dot = np.concatenate([x_dot, gyro_bias_dot])
        return x_dot

    def state_derivative_jac(self, x: np.ndarray) -> np.ndarray:
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
        q = x[6:10]
        w = x[10:13]
        dqdot_dq = 0.5 * left_q_3(q)
        dqdot_dw = 0.5 * right_q(np.concatenate([[0], w]))
        dwdot_dq = np.zeros((3, 4))
        dwdot_dw = self.I_sat_inv @ (skew(self.I_sat @ w) - skew(w) @ self.I_sat)

        jac = np.block(
            [
                [dv_dr, dv_dv, np.zeros((3, 7))],
                [da_dr, da_dv, np.zeros((3, 7))],
                [np.zeros((4, 6)), dqdot_dq, dqdot_dw],
                [np.zeros((3, 6)), dwdot_dq, dwdot_dw],
            ]
        )
        if self.has_gyro_bias:
            gyro_bias_jac = np.zeros((3, 3))
            # TODO: Implement gyro bias dynamics jacobian if needed
            jac = np.block(
                [
                    [jac, np.zeros((jac.shape[0], 3))],
                    [np.zeros((3, jac.shape[1])), gyro_bias_jac],
                ]
            )
        return jac

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

    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position dynamics under gravity.
        No perturbations are included.

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param dt: The amount of time between each time step.
        :return: A numpy array of shape (6,) containing the next state (position and velocity).
        """
        return Dynamics.RK4(x, self.state_derivative, dt)

    def f_jac(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position dynamics under gravity.
        No perturbations are included.

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param dt: The amount of time between each time step.
        :return: A numpy array of shape (6, 6) containing the state transition Jacobian.
        """
        return Dynamics.RK4_jac(x, self.state_derivative, self.state_derivative_jac, dt)

    def perturbed_state_derivative(self, x: np.ndarray, epoch: Epoch = None) -> np.ndarray:
        """
        The continuous-time state derivative function, dot{x} = f_c(x), for orbital position dynamics under gravity
        and the configured perturbations.

        :param x: A numpy array of shape (13,) containing the current state (position, velocity).
        :param epoch: The current time epoch. Can be None if the configured perturbations do not require it.

        :return: A numpy array of shape (13,) containing the full state derivative.
        """
        base_derivative = self.state_derivative(x)
        r = x[0:3]
        v = x[3:6]
        # q = x[6:10]
        # w = x[10:13]
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)

        updated_a = base_derivative[3:6]
        updated_w_dot = base_derivative[10:13]
        # Drag torque
        # if self.use_drag and drag_torque is not None:
        #     updated_w_dot += self.I_sat_inv @ drag_torque(r, v, q, t_J2000, self.drag_const, self.CoPM)

        # Gravity gradient torque
        # if gravity_gradient_torque is not None:
        #     updated_w_dot += self.I_sat_inv @ gravity_gradient_torque(r, self.I_sat)

        # Compute drag
        if self.use_drag and not np.isclose(v_norm, 0):
            if epoch is None:
                raise ValueError("Epoch is required to compute drag")
            a_drag_gt = drag_dynamics(x=x[0:6], drag_const=self.drag_const, latest_epoch=epoch)

            updated_a += a_drag_gt

        # Compute J2
        if self.use_j2 and not np.isclose(r_norm, 0):
            a_J2_gt = j2_dynamics(r)

            updated_a += a_J2_gt

        # Compute J3 and J4
        if self.use_j34 and not np.isclose(r_norm, 0):
            a_J3_gt = j3_dynamics(r)
            a_J4_gt = j4_dynamics(r)
            updated_a += a_J3_gt + a_J4_gt

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

        if self.has_gyro_bias:
            gyro_bias_dot = np.zeros(3)
            updated_w_dot = np.concatenate([updated_w_dot, gyro_bias_dot])

        return np.concatenate(
            [base_derivative[0:3], updated_a, base_derivative[6:10], updated_w_dot]
        )

    def perturbed_state_derivative_jac(self, x: np.ndarray, epoch: Epoch = None) -> np.ndarray:
        """
        The continuous-time state derivative Jacobian function, d(f_c)/dx, for orbital position dynamics under gravity
        and the configured perturbations.

        :param x: A numpy array of shape (6,) containing the current state (position, velocity).
        :param epoch: The current time epoch. Can be None if the configured perturbations do not require it.

        :return: A numpy array of shape (6,6) containing the state derivative Jacobian.
        """
        base_jacobian = self.state_derivative_jac(x)

        v = x[3:6]
        v_norm = np.linalg.norm(v)

        da_dr = base_jacobian[3:6, 0:3]  # pylint: disable=E1136  # pylint/issues/9590
        da_dv = base_jacobian[3:6, 3:6]  # pylint: disable=E1136  # pylint/issues/9590

        # Compute drag
        if self.use_drag and not np.isclose(v_norm, 0):
            if epoch is None:
                raise ValueError("Epoch is required to compute drag jacobian")
            da_drag_gt_dv = drag_jacobian(x=x[0:6], drag_const=self.drag_const, latest_epoch=epoch)

            da_dv += da_drag_gt_dv

        # Compute J2 either using autodiff or manually
        if self.use_j2 and not np.isclose(np.linalg.norm(x[0:3]), 0):
            da_J2_gt_dr = j2_jacobian_manual(x[:3])
            # da_J2_gt_dr = j2_jacobian_auto(x[0:3])

            da_dr += da_J2_gt_dr

        # Compute J3 and J4
        if self.use_j34 and not np.isclose(np.linalg.norm(x[0:3]), 0):
            da_J3_gt_dr = j3_jacobian_auto(x[0:3])
            da_J4_gt_dr = j4_jacobian_auto(x[0:3])

            da_dr += da_J3_gt_dr + da_J4_gt_dr

        # Compute third body gravity
        if self.use_sun_grav:
            if epoch is None:
                raise ValueError("Epoch is required to compute sun gravitational effects jacobian")
            da_sun_gt_dr = sun_gravity_jac(r_sat=x[0:3], epoch=epoch)

            da_dr += da_sun_gt_dr

        if self.use_moon_grav:
            if epoch is None:
                raise ValueError("Epoch is required to compute moon gravitational effects jacobian")
            da_moon_gt_dr = moon_gravity_jac(r_sat=x[0:3], epoch=epoch)

            da_dr += da_moon_gt_dr

        jac = np.block(
            [
                [base_jacobian[0:3, 0:]],  # pylint: disable=E1136  # pylint/issues/9590
                [
                    da_dr,
                    da_dv,
                    base_jacobian[3:6, 6:],  # pylint: disable=E1136  # pylint/issues/9590
                ],
                [base_jacobian[6:, :]],  # pylint: disable=E1136  # pylint/issues/9590
            ]
        )

        if self.has_gyro_bias:
            gyro_bias_jac = np.zeros((3, 3))
            jac = np.block(
                [
                    [jac, np.zeros((jac.shape[0], 3))],
                    [np.zeros((3, jac.shape[1])), gyro_bias_jac],
                ]
            )

        return jac

    def perturbed_state_derivative_wrapper(
        self, epoch: Epoch = None
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Wrapper to perturbed_state_derivative to avoid pylint error.
        """

        def ret(x: np.ndarray) -> np.ndarray:
            if self.require_epoch:
                return self.perturbed_state_derivative(x=x, epoch=epoch)
            return self.perturbed_state_derivative(x=x)

        return ret

    def perturbed_state_derivative_jac_wrapper(
        self, epoch: Epoch = None
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Wrapper to perturbed_state_derivative_jac to avoid pylint error.
        """

        def ret(x: np.ndarray) -> np.ndarray:
            if self.require_epoch:
                return self.perturbed_state_derivative_jac(x=x, epoch=epoch)
            return self.perturbed_state_derivative_jac(x=x)

        return ret

    def perturbed_f(self, x: np.ndarray, dt: float, epoch: Epoch = None) -> np.ndarray:
        """
        The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position dynamics under gravity
        and the configured perturbations.

        :param x: A numpy array of shape (13,) containing the current state (position, velocity).
        :param dt: The amount of time between each time step.
        :param epoch: The current time epoch. Can be None if the configured perturbations do not require it.

        :return: A numpy array of shape (13,) containing the next state (position, velocity).
        """
        func: Callable[[np.ndarray], np.ndarray] = self.perturbed_state_derivative_wrapper(
            epoch=epoch
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

        func: Callable[[np.ndarray], np.ndarray] = self.perturbed_state_derivative_wrapper(
            epoch=epoch
        )
        func_jac: Callable[[np.ndarray], np.ndarray] = self.perturbed_state_derivative_jac_wrapper(
            epoch=epoch
        )
        return Dynamics.RK4_jac(
            x=x,
            func=func,
            func_jac=func_jac,
            dt=dt,
        )

    def unmodelled_acceleration(self, x: np.ndarray, epoch: Epoch = None) -> np.ndarray:
        """
        Compute the acceleration terms not modelled in the EKF
        """
        r = x[0:3]
        # v = x[3:6]
        r_norm = np.linalg.norm(r)

        unmodelled_a = np.zeros(3)
        # Compute J2
        # if self.use_j2 and not np.isclose(r_norm, 0):
        #     a_J2_gt = j2_dynamics(r)
        #     unmodelled_a += a_J2_gt

        # Compute J3 and J4
        if self.use_j34 and not np.isclose(r_norm, 0):
            a_J3_gt = j3_dynamics(r)
            a_J4_gt = j4_dynamics(r)
            unmodelled_a += a_J3_gt + a_J4_gt

        # Compute third body gravity
        if self.use_sun_grav:
            if epoch is None:
                raise ValueError("Epoch is required to compute sun gravitational effects")
            a_sun_gt = sun_gravity(r_sat=x[0:3], epoch=epoch)

            unmodelled_a += a_sun_gt

        if self.use_moon_grav:
            if epoch is None:
                raise ValueError("Epoch is required to compute moon gravitational effects")
            a_moon_gt = moon_gravity(r_sat=x[0:3], epoch=epoch)

            unmodelled_a += a_moon_gt

        return np.array(unmodelled_a, dtype=np.float64)

    def drag_constant(self, x: np.ndarray, epoch: Epoch = None) -> float:
        """
        Get the drag constant for the dynamics.

        :return: The drag constant in m^2/kg.
        """
        height = np.linalg.norm(x[0:3]) - R_EARTH
        ekf_density = NOMINAL_DENSITY * np.exp(-height / REF_HEIGHT)
        true_density = density_harris_priester(x=x * 1e3, epoch=epoch)

        return np.array([true_density / ekf_density])

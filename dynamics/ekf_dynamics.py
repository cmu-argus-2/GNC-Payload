"""
Functions for implementing EKF dynamics extending the orbital dynamics.
"""

# pylint: disable=import-error
from functools import partial

import numpy as np
from brahe import Epoch
from brahe.constants import GM_EARTH

from dynamics.orbital_dynamics import Dynamics

# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments


class EKFDynamics(Dynamics):
    """
    This class contains the EKF dynamics functions. It inherits from the basic Dynamics class.
    """

    def __init__(
        self,
        config: dict,
        use_unmodelled_a: bool,
        use_drag: bool,
        use_j2: bool,
    ) -> None:
        """
        Initialize the EKFDynamics class.

        :param config: The configuration dictionary.
        :param use_unmodelled_a: Whether to use unmodelled accelerations in the dynamics.
        :param use_drag: Whether to use drag in the dynamics.
        :param use_j2: Whether to use J2 perturbations in the dynamics.
        :return: None
        """
        super().__init__(config=config, use_drag=use_drag, use_j2=use_j2)
        self.use_unmodelled_a = use_unmodelled_a

        if use_unmodelled_a:
            self.ua_std_dev = 1e-5

    def perturbed_state_derivative(self, x: np.ndarray, epoch: Epoch = None) -> np.ndarray:
        """
        The continuous-time state derivative function, dot{x} = f_c(x), for orbital position dynamics under gravity
        and the configured perturbations.

        :param x: A numpy array of shape (6,) or (9,) containing the current state position, velocity,
        (unmodelled_accelerations).
        :param epoch: The current time epoch. Can be None if the configured perturbations do not require it.

        :return: A numpy array of shape (6,) or (9,) containing the full state derivative.
        """

        base_derivative = super().perturbed_state_derivative(x[0:6], epoch)
        updated_a = base_derivative[3:6]

        # Compute unmodelled accelerations
        if self.use_unmodelled_a:
            unmodelled_a = x[6:9]
            ua_dot = np.random.normal(0, self.ua_std_dev, 3)

            updated_a += unmodelled_a

            return np.concatenate([base_derivative[0:3], updated_a, ua_dot])

        return base_derivative

    def perturbed_state_derivative_jac(self, x: np.ndarray, epoch: Epoch = None) -> np.ndarray:
        """
        The continuous-time state derivative Jacobian function, d(f_c)/dx, for orbital position dynamics under gravity
        and the configured perturbations.

        :param x: A numpy array of shape (6,) or (9,) containing the current state position, velocity,
        (unmodelled_accelerations).
        :param epoch: The current time epoch. Can be None if the configured perturbations do not require it.

        :return: A numpy array of shape (6,6) or (9,9) containing the state derivative Jacobian.
        """
        base_jacobian = super().perturbed_state_derivative_jac(x[0:6], epoch)

        # Compute unmodelled accelerations
        if self.use_unmodelled_a:
            dv_dua = np.zeros((3, 3))
            da_dua = np.eye(3)

            # Unmodelled accelerations have no partial derivatives itself so just
            # return a 3 x 9 matrix with zeros

            return np.block(
                [
                    [base_jacobian[0:3, 0:6], dv_dua],
                    [base_jacobian[3:6, 0:6], da_dua],
                    [np.zeros((3, 9))],
                ]
            )

        return base_jacobian

        # TODO: incorporate drag estimate into the jacobian
        # dest_drag = np.zeros((3,9))
        # dv_dest_drag = np.zeros((3,1))

        # da_dest_drag = self.drag_const * self.nominal_density * v / v_norm
        # dua_dest_drag = np.zeros((3,1))
        # dest_drag_dest_drag = np.eye((1))

    def perturbed_f(self, x: np.ndarray, dt: float, epoch: Epoch = None) -> np.ndarray:
        """
        The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position dynamics
        with second order effects.
        J2 perturbations and drag can be included.
        :param x: A numpy array of shape (9,) containing the current state (position, velocity and
        unmodelled accelerations).
        :param dt: The amount of time between each time step.
        :param epoch: The current time epoch.

        :return: A numpy array of shape (9,) containing the next state (position, velocity and
        unmodelled accelerations).
        """
        func = (
            partial(self.perturbed_state_derivative, epoch=epoch)
            if self.use_drag
            else self.perturbed_state_derivative
        )
        return Dynamics.RK4(x=x, func=func, dt=dt)

    def perturbed_f_jac(self, x: np.ndarray, dt: float, epoch: Epoch = None) -> np.ndarray:
        """
        The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position dynamics
        with second order effects.
        J2 perturbations and drag can be included.

        :param x: A numpy array of shape (9,) containing the current state (position and velocity).
        :param dt: The amount of time between each time step.
        :param epoch: The current time epoch.

        :return: A numpy array of shape (9, 9) containing the state transition Jacobian.
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

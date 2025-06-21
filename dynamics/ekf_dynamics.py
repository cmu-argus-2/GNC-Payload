"""
Functions for implementing EKF dynamics extending the orbital dynamics.
"""

# pylint: disable=import-error
import numpy as np
from brahe import Epoch

from dynamics.drag_dynamics import (
    da_dest_drag_derivative,
    dadrag_dr_partial,
    dadrag_dv_partial,
    drag_scalar_estimate,
)
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
        use_drag_scalar: bool,
        use_sun_grav: bool,
        use_moon_grav: bool,
        use_drag: bool,
        use_j2: bool,
        use_j34: bool,
        ua_scale: float = 1,
    ) -> None:
        """
        Initialize the EKFDynamics class.

        :param config: The configuration dictionary.
        :param use_unmodelled_a: Whether to use unmodelled accelerations in the dynamics.
        :param use_drag_scalar: Whether to use a scalar drag estimate.
        :param use_moon_grav: Whether to use the moon's gravity in the dynamics.
        :param use_sun_grav: Whether to use the sun's gravity in the dynamics.
        :param use_drag: Whether to use drag in the dynamics.
        :param use_j2: Whether to use J2 perturbations in the dynamics.
        :param use_j34: Whether to use J3 and J4 perturbations in the dynamics.
        :param ua_scale: The scale factor for unmodelled accelerations.
        :return: None
        """
        super().__init__(
            config=config,
            use_drag=use_drag,
            use_j2=use_j2,
            use_j34=use_j34,
            use_sun_grav=use_sun_grav,
            use_moon_grav=use_moon_grav,
        )
        self.use_unmodelled_a = use_unmodelled_a
        self.use_drag_scalar = use_drag_scalar

        # State dim at least position and velocity
        self.state_dim = 6

        if use_unmodelled_a:
            self.ua_scale = ua_scale
            self.state_dim += 3

        if use_drag_scalar:
            self.state_dim += 1

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
        remainder = np.zeros((0,))

        # Compute unmodelled accelerations
        if self.use_unmodelled_a:
            updated_a += x[6:9] / self.ua_scale
            remainder = np.append(remainder, np.zeros((3,)))

        if self.use_drag_scalar:
            drag_a = drag_scalar_estimate(x=x[0:6], d_est=x[9], drag_const=self.drag_const)
            updated_a += drag_a
            remainder = np.append(remainder, np.zeros((1,)))

        return np.concatenate([base_derivative[0:3], updated_a, remainder])

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

        if self.use_drag_scalar:
            dv_dest_drag = np.zeros((3, 1))
            da_dest_drag = da_dest_drag_derivative(x[0:6], self.drag_const)
            daestdrag_dr = dadrag_dr_partial(x=x[0:6], d_est=x[9], drag_const=self.drag_const)
            daestdrag_dv = dadrag_dv_partial(x=x[0:6], d_est=x[9], drag_const=self.drag_const)
            base_jacobian[3:6, 0:3] += daestdrag_dr  # pylint: disable=E1137
            base_jacobian[3:6, 3:6] += daestdrag_dv  # pylint: disable=E1137
            drag_jac = np.zeros((1, self.state_dim))
        else:
            dv_dest_drag = np.zeros((3, 0))
            da_dest_drag = np.zeros((3, 0))
            drag_jac = np.zeros((0, self.state_dim))

        # Compute unmodelled accelerations
        if self.use_unmodelled_a:
            dv_dua = np.zeros((3, 3))
            da_dua = np.eye(3)
            ua_jac = np.zeros((3, self.state_dim))
        else:
            dv_dua = np.zeros((3, 0))
            da_dua = np.zeros((3, 0))
            ua_jac = np.zeros((0, self.state_dim))
            # Unmodelled accelerations have no partial derivatives itself so just
            # return a 3 x 9 matrix with zeros

        return np.block(
            [
                [
                    base_jacobian[0:3, 0:6],  # pylint: disable=E1136  # pylint/issues/9590
                    dv_dua,
                    dv_dest_drag,
                ],
                [
                    base_jacobian[3:6, 0:6],  # pylint: disable=E1136  # pylint/issues/9590
                    da_dua,
                    da_dest_drag,
                ],
                [ua_jac],
                [drag_jac],
            ]
        )

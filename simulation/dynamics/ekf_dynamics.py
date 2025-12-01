"""
Functions for implementing EKF dynamics extending the orbital dynamics.
"""

# pylint: disable=import-error
import numpy as np
from brahe import R_EARTH, Epoch
from brahe.constants import GM_EARTH
from dynamics.drag_dynamics import (
    da_dest_drag_derivative,
    dadrag_dr_partial,
    dadrag_dv_partial,
    density_exponential,
    drag_scalar_estimate,
)
from dynamics.grav_potential_dynamics import j3_dynamics, j4_dynamics
from dynamics.orbital_dynamics import Dynamics
from dynamics.third_body_dynamics import moon_gravity, sun_gravity

from utils.earth_utils import density_harris_priester

# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=R0913
# too-many-positional-arguments
GM_EARTH = GM_EARTH / 1e9  # Convert to km^3/s^2
R_EARTH = R_EARTH / 1e3  # km

# Exponential model parameters from U.S. Standard Atmosphere 1976
# Taken from Fundamentals of Astrodynamics and Applications, 4th Edition, by David A. Vallado
H_ELLP = [300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 700.0]
NOMINAL_DENSITY = [2.418e-2, 9.518e-3, 3.725e-3, 1.585e-3, 6.967e-4, 1.454e-4]  # kg/km^3
SCALE_HEIGHT = [53.628, 53.298, 58.515, 60.828, 63.822, 71.835]  # km


class EKFDynamics(Dynamics):
    """
    This class contains the EKF dynamics functions. It inherits from the basic Dynamics class.
    """

    # pylint: disable=R0917
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
            updated_a += x[6:9]
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

    def true_unmodelled_acceleration(self, x: np.ndarray, epoch: Epoch = None) -> np.ndarray:
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
        if not self.use_j34 and not np.isclose(r_norm, 0):
            a_J3_gt = j3_dynamics(r)
            a_J4_gt = j4_dynamics(r)
            unmodelled_a += a_J3_gt + a_J4_gt

        # Compute third body gravity
        if not self.use_sun_grav:
            if epoch is None:
                raise ValueError("Epoch is required to compute sun gravitational effects")
            a_sun_gt = sun_gravity(r_sat=r, epoch=epoch)

            unmodelled_a += a_sun_gt

        if not self.use_moon_grav:
            if epoch is None:
                raise ValueError("Epoch is required to compute moon gravitational effects")
            a_moon_gt = moon_gravity(r_sat=r, epoch=epoch)

            unmodelled_a += a_moon_gt

        return np.array(unmodelled_a, dtype=np.float64)

    def true_drag_constant(self, x: np.ndarray, epoch: Epoch = None) -> float:
        """
        Get the drag constant for the dynamics.

        :return: The drag constant in m^2/kg.
        """
        ekf_density = density_exponential(x)
        true_density = density_harris_priester(x=x * 1e3, epoch=epoch)

        return np.array([true_density / ekf_density])

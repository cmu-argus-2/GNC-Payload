""" Extended Kalman Filter for orbit determination """

# pylint: disable=import-error
from typing import Any, List

import brahe
import jax
import jax.numpy as jnp
import numpy as np
import quaternion
from brahe import Epoch

from dynamics.orbital_dynamics import Dynamics
from sensors.camera_model import CameraModelManager
from utils.math_utils import der_rp2q, left_q, left_q_3, quat2rotm, rot_2_q  # right_q
from vision_inference.logger import Logger


# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=R0913
# too-many-positional-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=no-member
# pylint: disable=too-many-locals
# pylint: disable=E1136  # pylint/issues/9590
class EKF:
    """
    Extended Kalman Filter
    """

    # pylint: disable=R0917
    def __init__(
        self,
        r: np.ndarray,
        v: np.ndarray,
        q: Any,  # Should be of type numpy.quaternion but mypy doesn't seem to recognise it.
        # a_b: np.ndarray,
        w_b: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        dt: float,
        config: dict,
        ua: np.ndarray,
        ekf_dynamics: Dynamics,
        state_scaling: np.ndarray,
    ) -> None:
        """
        Initialize the EKF

        :param r: Initial position state expressed in inertial frame (ECEF) with shape (3,)
        :param v: Initial velocity state expressed in body frame with shape (3,)
        :param q: Initial attitude quaternion with shape (4,) where the scalar component is first.
            Note: The quaternion is of type numpy.quaternion, not np.ndarray and assumed to be normalized
        # :param a_b: Initial accelerometer bias with shape (3,)
        :param w_b: Initial angular velocity bias with shape (3,)
        :param P: Initial covariance with shape (9, 9)
        :param Q: Process noise covariance with shape (16, 16)
        :param dt: The amount of time between each time step.
        :param config: The configuration dictionary.
        :param ua: The unmodeled acceleration with shape (3,)
        :param ekf_dynamics: The Dynamics object used to calculate the dynamics of the system.
        :param state_scaling: The vector of scale factors for the states.

        :return: None

        """
        self.state_scaling = state_scaling

        self.r_m = r
        self.r_p = r

        self.v_m = v
        self.v_p = v

        self.q_m = q
        self.q_p = q

        # self.a_b = a_b
        self.w_b = w_b

        self.ua = ua
        self.drag_est = np.array([1])

        self.P_m = P * (self.state_scaling**2)
        self.P_p = P * (self.state_scaling**2)

        self.Q = Q * (self.state_scaling**2)
        self.R = np.zeros((3, 3))
        self.dt = dt

        self.cond_threshold = 1e15
        self.H = np.append(np.zeros((1, 3)), np.eye(3), axis=0)
        self.config = config
        self.ekf_dynamics = ekf_dynamics

    def predict(self, u: np.ndarray, epoch: Epoch = None) -> None:
        """
        Predict the next prior state. This corresponds to the prior update step in the EKF algorithm.
        Using Zac Manchester's formulation as defined in his inertial filter examples notebook
        https://github.com/RoboticExplorationLab/inertial-filter-examples

        :param u: IMU measurements consisting of angular velocity and linear acceleration with shape (6,)
        :param epoch: The epoch at which the prediction is made. If None is passed, no epoch is used.

        :return: None
        """

        # TODO: Use IMU measurements and update quaternion estimate

        w = u[0:3]  # angular velocity measurement from IMU

        x_unscaled = np.concatenate(
            [self.r_m, self.v_m, self.ua, self.drag_est, np.zeros(3), self.w_b]
        )
        # x_scaled     = x * self.state_scaling
        A_pos = self.ekf_dynamics.perturbed_f_jac(x=x_unscaled[:10], dt=self.dt, epoch=epoch)
        x_new_unscaled = self.ekf_dynamics.perturbed_f(x=x_unscaled[:10], dt=self.dt, epoch=epoch)
        # x_new_scaled   = x_new_unscaled * self.state_scaling
        self.q_p = left_q(self.q_m) @ quaternion.as_float_array(
            quaternion.from_rotation_vector(self.dt * (w - x_unscaled[13:]))
        )

        self.r_p = x_new_unscaled[0:3]
        self.v_p = x_new_unscaled[3:6]

        dqdq = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(-1 * self.dt * (w - x_unscaled[13:]))
        )
        dqdw = (
            -1
            * self.dt
            * left_q_3(self.q_p).T
            @ left_q(self.q_m)
            @ der_rp2q(self.dt * (w - x_unscaled[13:]))
        )

        A = np.block(
            [
                [A_pos, np.zeros((10, 6))],
                [np.zeros((3, 10)), dqdq, dqdw],
                [np.zeros((3, 13)), np.eye(3)],
            ]
        )
        A_scaled = np.diag(self.state_scaling) @ A @ np.diag(1.0 / self.state_scaling)

        self.P_p = A_scaled @ self.P_m @ A_scaled.T + self.Q

    def no_measurement(self) -> None:
        """
        If no measurements are taken, just take the prior state to be the posterior state.
        """
        self.r_m = self.r_p
        self.q_m = self.q_p
        self.v_m = self.v_p
        self.P_m = self.P_p

    # pylint: disable=R0917
    def measurement(
        self,
        z: List[np.ndarray],
        camera_model_manager: CameraModelManager,
        measurement_camera_names: np.ndarray,
        epoch: Epoch,
        num_iter: int = 1,
    ) -> None:
        """
        Update the state estimate based on the measurement. This corresponds to the posterior update step
        in the EKF algorithm.

        :param z: Measurement consisting of a tuple of the bearing unit vectors in the body frame and the
        landmark positions in ECI coordinates, both with shape (N, 3)
        :param camera_model_manager: The camera model manager used to manage the cameras.
        :param measurement_camera_names: The names of the cameras that took the measurements.
        :param epoch: The epoch at which the measurement is made. Epoch must be provided for ecef-eci transformation!
        :param num_iter: Number of iterations of the update steps to perform. Default is 1.

        :return: None
        """
        # Select a random fraction of the measurements to use to speed up computations
        mask = np.random.choice([True, False], size=z[0].shape[0], p=[0.04, 0.96])
        z0 = z[0][mask]
        z1 = z[1][mask]

        measurement_camera_names = measurement_camera_names[mask]

        # Flatten the measurement vector
        z0 = np.array(z0.reshape(-1))

        # Chance that the measurement vector is empty when mask is applied
        # (higher likelihood with fewer measurements)
        if z0.shape[0] == 0:
            self.no_measurement()
            Logger.log("INFO", "No measurements taken")
            return

        # Let R take the dimensionality of the number of measurements
        self.R = np.diag([1e-2] * z0.shape[0])

        x_p = jnp.array(  # unscaled
            np.concatenate(
                [
                    self.r_p,
                    self.v_p,
                    self.ua,
                    self.drag_est,
                    quaternion.as_rotation_vector(quaternion.as_quat_array(self.q_p)),
                    self.w_b,
                ]
            )
        )
        # Iterated Update
        for i in range(num_iter):

            h = self.h_est(z1, camera_model_manager, measurement_camera_names, x_p, epoch=epoch)
            H_unscaled = self.h_jac(
                z1, camera_model_manager, measurement_camera_names, x_p, epoch=epoch
            )
            H = H_unscaled @ np.diag(1.0 / self.state_scaling)
            S = H @ self.P_p @ H.T + self.R

            # Check for ill-conditioned matrix and add regularization if necessary
            if i == 0:
                cond = np.linalg.cond(S)
                if cond > self.cond_threshold:
                    S += np.eye(S.shape[0]) * 1e-6
                    Logger.log(
                        "WARNING", "Ill-conditioned matrix detected. Regularization applied."
                    )

            K = self.P_p @ H.T @ np.linalg.inv(S)

            delta_scaled = K @ (z0 - h)
            delta = delta_scaled / self.state_scaling

            self.r_m = np.array(x_p[0:3]) + delta[0:3]
            self.v_m = np.array(x_p[3:6]) + delta[3:6]
            self.ua = np.array(x_p[6:9]) + delta[6:9]
            self.drag_est = np.array(x_p[9:10]) + delta[9:10]
            self.q_m = quaternion.as_rotation_vector(
                quaternion.from_rotation_vector(np.array(x_p[10:13]))
                * quaternion.from_rotation_vector(delta[10:13])
            )
            self.w_b = np.array(x_p[13:16]) + delta[13:16]

            # Joseph form covariance update
            self.P_m = (np.eye(self.P_m.shape[0]) - K @ H) @ self.P_p @ (
                np.eye(self.P_m.shape[0]) - K @ H
            ).T + K @ self.R @ K.T

            x_p = jnp.array(
                np.concatenate([self.r_m, self.v_m, self.ua, self.drag_est, self.q_m, self.w_b])
            )
        # Convert final iterated rotation vector to quaternion
        self.q_m = quaternion.as_float_array(quaternion.from_rotation_vector(self.q_m))

    # pylint: disable=R0917
    def h_jac(
        self,
        z: np.ndarray,
        camera_model_manager: CameraModelManager,
        measurement_camera_names: np.ndarray,
        x_p: jnp.ndarray,
        epoch: Epoch,
    ) -> jnp.ndarray:
        """
        Calculate the Jacobian of the measurement model with respect to the state.

        :param z: Measurement consisting of the landmark locations in ECI coordinates with shape (N, 3)
        :param camera_model_manager: The camera model manager used to manage the cameras.
        :param measurement_camera_names: Array of names of the cameras that took each measurement.
        :param x_p: Prior state estimate consisting of position, quaternion and velocity with shape (9,)
        :param epoch: The epoch at which the measurement is made. Epoch must be provided for ecef-eci transformation!

        :return: The Jacobian of the measurement model with respect to the state.
        """
        jac = jax.jacobian(self.h_est, argnums=3)(
            z, camera_model_manager, measurement_camera_names, x_p, epoch=epoch
        )

        return jac

    # pylint: disable=R0917
    def h_est(
        self,
        z: np.ndarray,
        camera_model_manager: CameraModelManager,
        measurement_camera_names: np.ndarray,
        x_p: jnp.ndarray,
        epoch: Epoch,
    ) -> jnp.ndarray:
        """
        Generate an estimate from measurements made. Using the known locations of the landmarks, we can provide
        a bearing estimate.

        :param z: Measurements of the landmarks in frame, consisting of just the ECI coordinates of the landmarks
        with shape (N, 3)
        :param camera_model_manager: The camera model manager used to manage the cameras.
        :param measurement_camera_names: Array of names of the cameras that took each measurement.
        :param x_p: Prior state estimate consisting of [position, velocity, rotation_vector] with shape (9,)
        :param epoch: The epoch at which the measurement is made. Epoch must be provided for ecef-eci transformation!

        :return: Estimate of the bearing vectors to all landmarks in the body frame with shape (N * 3, )
        """
        estimate = jnp.zeros((len(z) * 3))

        # Define rotation matrices
        # transform rotation_vector to rotation matrix via quaternion
        eci_R_body = quat2rotm(rot_2_q(x_p[10:13]))
        ecef_R_eci = brahe.frames.rECItoECEF(epc=epoch)
        ecef_R_body = ecef_R_eci @ eci_R_body

        # Transform landmarks and position from ECI to ECEF
        landmarks_ecef = (ecef_R_eci @ z.T).T
        position_ecef = ecef_R_eci @ x_p[0:3]

        # Assert landmarks and measurement camera names are the same length
        assert landmarks_ecef.shape[0] == len(
            measurement_camera_names
        ), "Landmarks and measurement camera names must be the same length"

        # Calculate estimated bearing unit vectors in ECEF and transform to body frame
        for i, land_pos_ecef in enumerate(landmarks_ecef):
            # account for camera position in ECEF
            camera_position_ecef = camera_model_manager[
                measurement_camera_names[i]
            ].get_camera_position(position_ecef, ecef_R_body)

            vec_ecef = land_pos_ecef - camera_position_ecef
            vec_ecef /= jnp.linalg.norm(vec_ecef)
            body_vec = ecef_R_body.T @ vec_ecef
            estimate = estimate.at[i * 3 : i * 3 + 3].set(body_vec)

        return estimate

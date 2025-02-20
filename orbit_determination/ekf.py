""" Extended Kalman Filter for orbit determination """

import math
from typing import Any, Tuple

import brahe
import jax
import jax.numpy as jnp
import numpy as np
import quaternion

from dynamics.orbital_dynamics import f, f_jac
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from utils.math_utils import R, left_q, rot_2_q  # right_q

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=no-member


class EKF:
    """
    Extended Kalman Filter
    """

    def __init__(
        self,
        r: np.ndarray,
        v: np.ndarray,
        q: Any,  # Should be of type numpy.quaternion but mypy doesn't seem to recognise it.
        # a_b: np.ndarray,
        # w_b: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        R_vec: np.ndarray,
        dt: float,
        config: dict,
        w: np.ndarray,
    ) -> None:
        """
        Initialize the EKF

        :param r: Initial position state expressed in inertial frame (ECEF) with shape (3,)
        :param v: Initial velocity state expressed in body frame with shape (3,)
        :param q: Initial attitude quaternion with shape (4,) where the scalar component is first.
            Note: The quaternion is of type numpy.quaternion, not np.ndarray and assumed to be normalized
        # :param a_b: Initial accelerometer bias with shape (3,)
        # :param w_b: Initial angular velocity bias with shape (3,)
        :param P: Initial covariance with shape (9, 9)
        :param Q: Process noise covariance with shape (16, 16)
        :param R_vec: Measurement noise covariance with shape depending on the number of landmarks
        :param dt: The amount of time between each time step.
        :param w: The angular velocity of the satellite with shape (3,)

        :return: None

        Note on R_vec matrix dimensionality: As the number of landmarks observed will change between
        individual time steps, the R matrix needs to be constructed at each time step where the vision
        pipeline is used. The dimensionality of the matrix is 3n x 3n where n is the number of landmarks
        observed.
        """

        self.r_m = r
        self.r_p = r

        self.v_m = v
        self.v_p = v

        self.q_m = q
        self.q_p = q

        # self.a_b = a_b
        # self.w_b = w_b

        # Scale the attitude Covariance
        P[6:9, 6:9] *= (1e-9,)

        self.P_m = P
        self.P_p = P

        self.w = w

        self.Q = Q
        self.R = R_vec
        self.dt = dt

        camera_params = config["satellite"]["camera"]
        self.t_body_to_camera = np.asarray(camera_params["t_body_to_camera"])

        self.cond_threshold = 1e15
        self.H = np.append(np.zeros((1, 3)), np.eye(3), axis=0)

    def predict(self, u: np.ndarray) -> None:
        """
        Predict the next prior state. This corresponds to the prior update step in the EKF algorithm.
        Using Zac Manchester's formulation as defined in his inertial filter examples notebook
        https://github.com/RoboticExplorationLab/inertial-filter-examples

        :param u: IMU measurements consisting of angular velocity and linear acceleration with shape (6,)

        :return: None
        """

        # TODO: Use IMU measurements and update quaternion estimate

        w = u[0:3]  # angular velocity measurement from IMU
        # self.r_p = self.r_m + self.dt * self.v_m
        # self.q_p = self.q_m * quaternion.from_rotation_vector(0.5 * self.dt * wf)
        # self.v_p = self.v_m + self.dt * (-GM_EARTH / np.linalg.norm(self.r_m) ** 3) * self.r_m

        x = np.concatenate([self.r_m, self.v_m])
        A_pos = f_jac(x, self.dt)
        x_new = f(x, self.dt)

        self.q_p = left_q(self.q_m) @ quaternion.as_float_array(
            quaternion.from_rotation_vector(0.5 * self.dt * w)
        )

        self.r_p = x_new[0:3]
        self.v_p = x_new[3:6]

        # A_att = self.H.T @ left_q(self.q_p).T @ left_q(self.q_m) @ right_q(quaternion.as_float_array(
        # quaternion.from_rotation_vector(self.w))) @ self.H
        A_att = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(-0.5 * self.dt * w))

        A = np.block([[A_pos, np.zeros((6, 3))], [np.zeros((3, 6)), A_att]])

        self.P_p = A @ self.P_m @ A.T + self.Q

    def no_measurement(self) -> None:
        """
        If no measurements are taken, just take the prior state to be the posterior state.
        """
        self.r_m = self.r_p
        self.q_m = self.q_p
        self.v_m = self.v_p
        self.P_m = self.P_p

    def measurement(
        self,
        z: Tuple[np.ndarray, np.ndarray],
        body_Rs_camera: np.ndarray,
        data_manager: ODSimulationDataManager,
        num_iter: int = 1,
    ) -> None:
        """
        Update the state estimate based on the measurement. This corresponds to the posterior update step
        in the EKF algorithm.

        :param z: Measurement consisting of a tuple of the bearing unit vectors in the body frame and the
        landmark positions in ECI coordinates, both with shape (N, 3)
        :param body_Rs_camera: Rotation matrices from the camera frame to the body frame with shape (N, 3, 3)
        :param data_manager: The ODSimulationDataManager object containing the simulation data.
        :param num_iter: Number of iterations of the update steps to perform. Default is 1.

        :return: None
        """
        # Select a fraction of the measurements to use to speed up computations
        z0 = z[0][: int(math.ceil(z[0].shape[0] * 0.05))]
        z1 = z[1][: int(math.ceil(z[1].shape[0] * 0.05))]

        # Flatten the measurement vector
        z0 = np.array(z0.reshape(-1))

        # Let R take the dimensionality of the number of measurements
        self.R = np.diag([1e-5] * z0.shape[0])

        x_p = jnp.array(
            np.concatenate(
                [
                    self.r_p,
                    self.v_p,
                    quaternion.as_rotation_vector(quaternion.as_quat_array(self.q_p)),
                ]
            )
        )
        # Iterated Update
        for i in range(num_iter):

            h = self.h_est(z1, data_manager, x_p)
            H = self.h_jac(z1, data_manager, x_p)
            S = H @ self.P_p @ H.T + self.R

            # Check for ill-conditioned matrix and add regularization if necessary
            if i == 0:
                cond = np.linalg.cond(S)
                if cond > self.cond_threshold:
                    S += np.eye(S.shape[0]) * 1e-6
                    print("Ill-conditioned matrix detected. Regularization applied.")

            K = self.P_p @ H.T @ np.linalg.inv(S)

            delta = K @ (z0 - h)

            self.r_m = np.array(x_p[0:3]) + delta[0:3]
            self.v_m = np.array(x_p[3:6]) + delta[3:6]
            self.q_m = quaternion.as_rotation_vector(
                quaternion.from_rotation_vector(np.array(x_p[6:9]))
                * quaternion.from_rotation_vector(delta[6:9])
            )

            # Joseph form covariance update
            self.P_m = (np.eye(self.P_m.shape[0]) - K @ H) @ self.P_p @ (
                np.eye(self.P_m.shape[0]) - K @ H
            ).T + K @ self.R @ K.T

            x_p = jnp.array(np.concatenate([self.r_m, self.v_m, self.q_m]))
        # Convert final iterated rotation vector to quaternion
        self.q_m = quaternion.as_float_array(quaternion.from_rotation_vector(self.q_m))

    def h_jac(
        self, z: np.ndarray, data_manager: ODSimulationDataManager, x_p: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the Jacobian of the measurement model with respect to the state.

        :param z: Measurement consisting of the landmark locations in ECI coordinates.
        :param data_manager: The ODSimulationDataManager object containing the simulation data.
        :param x_p: Prior state estimate consisting of position, quaternion and velocity with shape (9,)

        :return: The Jacobian of the measurement model with respect to the state.
        """
        jac = jax.jacobian(self.h_est, argnums=2)(z, data_manager, x_p)

        return jac

    def h_est(
        self, z: np.ndarray, data_manager: ODSimulationDataManager, x_p: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Generate an estimate from measurements made. Using the known locations of the landmarks, we can provide
        a bearing estimate.

        :param z: Measurements of the landmarks in frame, consisting of just the ECI coordinates of the landmarks
        with shape (N, 3)
        :param data_manager: The ODSimulationDataManager object containing the simulation data.
        :param x_p: Prior state estimate consisting of [position, velocity, rotation_vector] with shape (9,)

        :return: Estimate of the bearing vectors to all landmarks in the body frame with shape (N * 3, )
        """
        estimate = jnp.zeros((len(z) * 3))

        # Define rotation matrices
        # transform rotation_vector to rotation matrix via quaternion
        eci_R_body = R(rot_2_q(x_p[6:9]))
        ecef_R_eci = brahe.frames.rECItoECEF(data_manager.latest_epoch)
        ecef_R_body = ecef_R_eci @ eci_R_body

        # Transform landmarks and position from ECI to ECEF
        landmarks_ecef = (ecef_R_eci @ z.T).T
        position_ecef = ecef_R_eci @ x_p[0:3] + ecef_R_body @ self.t_body_to_camera

        # Calculate estimated bearing unit vectors in ECEF and transform to body frame
        for i, land_pos_ecef in enumerate(landmarks_ecef):
            vec_ecef = land_pos_ecef - position_ecef
            vec_ecef /= jnp.linalg.norm(vec_ecef)
            body_vec = ecef_R_body.T @ vec_ecef
            estimate = estimate.at[i * 3 : i * 3 + 3].set(body_vec)

        return estimate

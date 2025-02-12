""" Extended Kalman Filter for orbit determination """

import math
from typing import Any, Tuple

import brahe
import jax
import jax.numpy as jnp
import numpy as np
import quaternion
import yaml
from brahe.constants import GM_EARTH
from scipy.spatial.transform import Rotation

from dynamics.orbital_dynamics import f, f_jac
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from utils.math_utils import R, rot_2_q


class EKF:
    """
    Extended Kalman Filter
    """

    def __init__(
        self,
        r: np.ndarray,
        v: np.ndarray,
        q: Any, # Should be of type numpy.quaternion but mypy doesn't seem to recognise it.
        # a_b: np.ndarray,
        # w_b: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        dt: float,
        config: dict,
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
        :param R: Measurement noise covariance with shape (3, 3)
        :param dt: The amount of time between each time step.

        :return: None

        Note on R matrix dimensionality: As the number of landmarks observed will change between individual time steps, 
        the R matrix needs to be constructed at each time step where the vision pipeline is used.
        """

        self.r_m = r
        self.r_p = r

        self.v_m = v
        self.v_p = v

        self.q_m = q
        self.q_p = q

        # self.a_b = a_b
        # self.w_b = w_b

        self.P_m = P
        self.P_p = P

        self.Q = Q
        self.R = R
        self.dt = dt

        camera_params = config["satellite"]["camera"]
        self.t_body_to_camera = np.asarray(camera_params["t_body_to_camera"])

        self.cond_threshold = 1e15

    def predict(self, u: np.ndarray) -> None:
        """
        Predict the next prior state. This corresponds to the prior update step in the EKF algorithm.
        Using Zac Manchester's formulation as defined in his inertial filter examples notebook
        https://github.com/RoboticExplorationLab/inertial-filter-examples

        :param u: IMU measurements consisting of angular velocity and linear acceleration with shape (6,)

        :return: None
        """

        # TODO: Use IMU measurements and update quaternion estimate
        # Using RK4 TODO: Add quaternion dynamics into orbit_dynamics.py so we can use the f and f_jac functions

        # wf = u[0:3]  # angular velocity measurement from IMU
        # self.r_p = self.r_m + self.dt * self.v_m
        # self.q_p = self.q_m * quaternion.from_rotation_vector(0.5 * self.dt * wf)
        # self.v_p = self.v_m + self.dt * (-GM_EARTH / np.linalg.norm(self.r_m) ** 3) * self.r_m

        # A = f_jac(self.x_m, self.dt) # TODO: Fix to include attitude

        # Jacobian
        # dqdq = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(0.5 * self.dt * wf))
        # dadq is zero as velocity in inertial frame not coupled to quaternion

        # A = np.block(
        #     [
        #         [np.eye(3), np.zeros((3, 3)), np.eye(3) * self.dt],
        #         [np.zeros((3, 3)), dqdq, np.zeros((3, 3))],
        #         [dadr, np.zeros((3, 3)), np.eye(3)],
        #     ]
        # )
        # A = np.block(
        #     [
        #         [np.eye(3), self.dt * np.eye(3)],
        #         [dadr, np.eye(3)],
        #     ]
        # )
        x = np.append(self.r_m, self.v_m)
        A = f_jac(x, self.dt)
        x_new = f(x, self.dt)
        self.r_p = x_new[0:3]
        self.v_p = x_new[3:6]

        self.P_p = A @ self.P_m @ A.T + self.Q

    def no_measurement(self) -> None:
        """
        If no measurements are taken, just take the prior state to be the posterior state.
        """
        self.r_m = self.r_p
        # self.q_m = self.q_p
        self.v_m = self.v_p
        self.P_m = self.P_p

    def measurement(
        self, z: Tuple[np.ndarray, np.ndarray], data_manager: ODSimulationDataManager
    ) -> None:
        """
        Update the state estimate based on the measurement. This corresponds to the posterior update step
        in the EKF algorithm.

        :param z: Measurement consisting of a tuple of the bearing unit vectors in the body frame and the
        landmark positions in ECI coordinates with shape (N, 3)
        :param data_manager: The ODSimulationDataManager object containing the simulation data.

        :return: None
        """
        # x_p = jnp.array(
        #     np.concatenate((self.r_p, quaternion.as_rotation_vector(self.q_p), self.v_p), axis=0)
        # )

        x_p = jnp.array(np.concatenate((self.r_p, self.v_p), axis=0))

        # Select a fraction of the measurements to use to speed up computations
        z0 = z[0][: int(math.ceil(z[0].shape[0] * 0.01))]
        z1 = z[1][: int(math.ceil(z[1].shape[0] * 0.01))]

        h = self.h(z1, data_manager, x_p)
        H = self.H(z1, data_manager, x_p)

        # z = (z0, z1)
        z0 = np.array(z0.reshape(-1))  # Flatten the measurement vector and cast to jax array

        # Let R take the dimensionality of the number of measurements
        self.R = np.diag([1e-5] * z0.shape[0])

        S = H @ self.P_p @ H.T + self.R

        cond = np.linalg.cond(S)

        # Check for ill-conditioned matrix and add regularization if necessary
        if cond > self.cond_threshold:
            S += np.eye(S.shape[0]) * 1e-6
            print("Ill-conditioned matrix detected. Regularization applied.")

        K = self.P_p @ H.T @ np.linalg.inv(S)

        delta = K @ (z0 - h)

        self.r_m = self.r_p + delta[0:3]
        # self.q_m = self.q_p * quaternion.from_rotation_vector(delta[3:6])
        self.v_m = self.v_p + delta[3:6]

        self.P_m = (np.eye(self.P_m.shape[0]) - K @ H) @ self.P_p @ (
            np.eye(self.P_m.shape[0]) - K @ H
        ).T + K @ self.R @ K.T  # Joseph form covariance update

    def H(
        self, z: np.ndarray, data_manager: ODSimulationDataManager, x_p: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the Jacobian of the measurement model with respect to the state.

        :param z: Measurement consisting of the landmark locations in ECI coordinates.
        :param data_manager: The ODSimulationDataManager object containing the simulation data.
        :param x_p: Prior state estimate consisting of position, quaternion and velocity with shape (10,)

        :return: The Jacobian of the measurement model with respect to the state.
        """
        jac = jax.jacobian(self.h, argnums=2)(z, data_manager, x_p)
        # J = np.zeros((len(z) * 3, 6))
        # for i, land_pos in enumerate(z):
        #     deriv = np.eye(3)/np.linalg.norm(x_p[0:3] - land_pos) - np.outer(x_p[0:3] - land_pos, x_p[0:3] /
        #   - land_pos)/(np.linalg.norm(x_p[0:3] - land_pos)**3)

        return jac

    def h(
        self, z: np.ndarray, data_manager: ODSimulationDataManager, x_p: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Generate an estimate from measurements made. Using the known locations of the landmarks, we can provide
        a bearing estimate.

        :param z: Measurements of the landmarks in frame, consisting of just the ECI coordinates of the landmarks
        with shape (N, 3)
        :param data_manager: The ODSimulationDataManager object containing the simulation data.
        :param x_p: Prior state estimate consisting of position, quaternion and velocity with shape (10,)

        :return: Estimate of the bearing vectors to all landmarks in the body frame with shape (N * 3, )
        """
        estimate = jnp.zeros((len(z) * 3))

        # Define rotation matrices
        t_idx = data_manager.state_count - 1
        eci_R_body = data_manager.eci_Rs_body[t_idx, ...]  # TODO: Fix using attitude state
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
        # for i, land_pos in enumerate(z):
        #     vec = x_p[:3] - land_pos
        #     vec /= jnp.linalg.norm(vec)
        #     # body_R_eci = R(rot_2_q(x_p[3:6]))
        #     # body_vec = body_R_eci @ vec
        #     estimate = estimate.at[i * 3 : i * 3 + 3].set(vec)

        return estimate

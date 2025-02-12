""" Extended Kalman Filter for orbit determination """

import math
import os
import sys
from typing import Any, Tuple

import brahe
import jax
import jax.numpy as jnp
import numpy as np
import quaternion
import yaml
from brahe.constants import GM_EARTH
from scipy.spatial.transform import Rotation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynamics.orbital_dynamics import f, f_jac
from utils.math_utils import R, rot_2_q


class EKF:
    """
    Extended Kalman Filter
    """

    def __init__(
        self,
        r: np.ndarray,
        v: np.ndarray,
        q: np.quaternion,
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
        """

        """
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
        self.R_camera_to_body = np.asarray(camera_params["R_camera_to_body"])
        self.t_body_to_camera = np.asarray(camera_params["t_body_to_camera"])

        self.cond_threshold = 1e15

    def predict(self, u: np.ndarray) -> None:
        """
        Predict the next prior state. This corresponds to the prior update step in the EKF algorithm.
        Using Zac Manchester's formulation as defined in his inertial filter examples notebook
        https://github.com/RoboticExplorationLab/inertial-filter-examples
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
        self.r_m = self.r_p
        # self.q_m = self.q_p
        self.v_m = self.v_p
        self.P_m = self.P_p

    def measurement(
        self, z: Tuple[np.ndarray, np.ndarray], landmark_bearing_sensor, data_manager
    ) -> None:
        """
        Update the state estimate based on the measurement. This corresponds to the posterior update step in the EKF algorithm.

        :param z: Measurement consisting of a tuple of the bearing unit vectors in the body frame and the landmark positions in ECI coordinates with shape (N, 3)
        """
        # x_p = jnp.array(
        #     np.concatenate((self.r_p, quaternion.as_rotation_vector(self.q_p), self.v_p), axis=0)
        # )

        x_p = jnp.array(np.concatenate((self.r_p, self.v_p), axis=0))

        # Select a fraction of the measurements to use to speed up computations
        z0 = z[0][: int(math.ceil(z[0].shape[0] * 0.01))]
        z1 = z[1][: int(math.ceil(z[1].shape[0] * 0.01))]

        z = (z0, z1)

        h = self.h(z[1], data_manager, x_p)
        H = self.H(z[1], data_manager, x_p)

        z = z[0].reshape(-1)  # Flatten the measurement vector

        # Let R take the dimensionality of the number of measurements
        self.R = np.diag([1e-5] * z.shape[0])

        S = H @ self.P_p @ H.T + self.R

        cond = np.linalg.cond(S)

        # Check for ill-conditioned matrix and add regularization if necessary
        if cond > self.cond_threshold:
            S += np.eye(S.shape[0]) * 1e-6
            print("Ill-conditioned matrix detected. Regularization applied.")

        K = self.P_p @ H.T @ np.linalg.inv(S)

        delta = K @ (z - h)

        self.r_m = self.r_p + delta[0:3]
        # self.q_m = self.q_p * quaternion.from_rotation_vector(delta[3:6])
        self.v_m = self.v_p + delta[3:6]

        self.P_m = (np.eye(self.P_m.shape[0]) - K @ H) @ self.P_p @ (
            np.eye(self.P_m.shape[0]) - K @ H
        ).T + K @ self.R @ K.T  # Joseph form covariance update

    def H(self, z: np.ndarray, data_manager, x_p: jnp.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian of the measurement model with respect to the state.

        :param z: Measurement consisting of the landmark locations in ECI coordinates.
        :param x_p: Prior state estimate consisting of position, quaternion and velocity with shape (10,)

        :return: The Jacobian of the measurement model with respect to the state.
        """
        jac = jax.jacobian(self.h, argnums=2)(z, data_manager, x_p)
        # J = np.zeros((len(z) * 3, 6))
        # for i, land_pos in enumerate(z):
        #     deriv = np.eye(3)/np.linalg.norm(x_p[0:3] - land_pos) - np.outer(x_p[0:3] - land_pos, x_p[0:3] - land_pos)/(np.linalg.norm(x_p[0:3] - land_pos)**3)

        return jac

    def h(self, z: np.ndarray, data_manager, x_p: np.ndarray) -> np.ndarray:
        """
        Generate an estimate from measurements made. Using the known locations of the landmarks, we can provide a bearing estimate.

        :param z: Measurements of the landmarks in frame, consisting of just the ECI coordinates of the landmarks with shape (N, 3)
        :param x_p: Prior state estimate consisting of position, quaternion and velocity with shape (10,)

        :return: Estimate of the bearing vectors to all landmarks in the body frame with shape (N * 3, )
        """
        """
        TODO:
        Transform landmark positions from ECI to ECEF
        Transform r_p from ECI to ECEF
        Calculate bearing vectors in ECEF
        Transform bearing vectors from ECEF to body frame
        """
        estimate = jnp.zeros((len(z) * 3))

        # Define rotation matrices
        t_idx = data_manager.state_count - 1
        R_body_to_eci = data_manager.Rs_body_to_eci[t_idx, ...]  # TODO: Fix using attitude state
        R_eci_to_ecef = brahe.frames.rECItoECEF(data_manager.latest_epoch)
        R_body_to_ecef = R_eci_to_ecef @ R_body_to_eci
        R_ecef_to_body = R_body_to_ecef.T

        # Transform landmarks and position from ECI to ECEF
        landmarks_ecef = (R_eci_to_ecef @ z.T).T
        position_ecef = R_eci_to_ecef @ x_p[0:3] + R_body_to_ecef @ self.t_body_to_camera

        # Calculate estimated bearing unit vectors in ECEF and transform to body frame
        for i, land_pos in enumerate(landmarks_ecef):
            vec = land_pos - position_ecef
            vec /= jnp.linalg.norm(vec)
            body_vec = R_ecef_to_body @ vec
            estimate = estimate.at[i * 3 : i * 3 + 3].set(body_vec)
        # for i, land_pos in enumerate(z):
        #     vec = x_p[:3] - land_pos
        #     vec /= jnp.linalg.norm(vec)
        #     # body_R_eci = R(rot_2_q(x_p[3:6]))
        #     # body_vec = body_R_eci @ vec
        #     estimate = estimate.at[i * 3 : i * 3 + 3].set(vec)

        return estimate


def load_config() -> dict[str, Any]:
    """
    Load the configuration file

    :return: The modified configuration file as a dictionary.
    """
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # decrease world update rate since we only care about position dynamics
    config["solver"]["world_update_rate"] = 1 / 60  # Hz
    config["mission"]["duration"] = 3 * 90 * 60  # s, roughly 1 orbit

    return config

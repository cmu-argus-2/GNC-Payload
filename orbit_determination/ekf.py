from typing import Any, Tuple

from scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dynamics.orbital_dynamics import f_jac
from vision_inference.landmark_bearings import LandmarkBearingSensor


class EKF:
    """
    Extended Kalman Filter
    """

    def __init__(
        self,
        x: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        att_quat: np.ndarray,
        dt: float,
    ) -> None:
        """
        Initialize the EKF

        :param x: Initial state consisting of position and velocity with respect to the ECI frame with shape (6,)
        :param P: Initial covariance with shape (6, 6)
        :param Q: Process noise covariance with shape (6, 6)
        :param R: Measurement noise covariance with shape (3, 3)
        :param att_quat: Initial attitude quaternion with shape (4,) where the scalar component is first.
        :param dt: The amount of time between each time step.
        """
        self.x_m = x
        self.x_p = x
        self.att_quat = att_quat  # TODO: Merge with MEKF to update attitude quaternion
        self.P_m = P
        self.P_p = P
        self.Q = Q
        self.R = R
        self.dt = dt

        self.cond_threshold = 1e15

        config = load_config()

        # set up landmark bearing sensor and orbit determination objects
        self.landmark_bearing_sensor = LandmarkBearingSensor(config)
        # TODO: Get the actual images into the EKF.

    def predict(self) -> None:
        """
        Predict the next prior state. This corresponds to the prior update step in the EKF algorithm.
        """
        A = f_jac(self.x_m, self.dt)
        self.x_p = A @ self.x_m
        self.P_p = A @ self.P_m @ A.T + self.Q

    def measurement(self, z: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Update the state estimate based on the measurement. This corresponds to the posterior update step in the EKF algorithm.

        :param z: Measurement consisting of a tuple of the bearing unit vectors in the body frame and the landmark positions in ECI coordinates.
        """

        h = self.h(z[1], self.x_p)
        H = self.H(z[1], self.x_p)

        S = H @ self.P_p @ H.T + self.R
        cond = np.linalg.cond(S)

        # Check for ill-conditioned matrix and add regularization if necessary
        if cond > self.cond_threshold:
            S += np.eye(S.shape[0]) * 1e-6

        K = self.P_p @ H.T @ np.linalg.inv(S)
        self.x_m = self.x_p + K @ (z - h)
        self.P_m = (np.eye(self.P_m.shape[0]) - K @ H) @ self.P_p @ (
            np.eye(self.P_m.shape[0]) - K @ H
        ).T + K @ self.R @ K.T  # Joseph form covariance update

    def H(self, z: np.ndarray, x_p: np.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian of the measurement model with respect to the state.

        :param z: Measurement consisting of the landmark locations in ECI coordinates.
        :param x_p: Prior state estimate consisting of position and velocity.
        :return: The Jacobian of the measurement model with respect to the state.
        """

        jac = jax.jacobian(self.h, argnums=1)(z, x_p)
        return jac

    def h(self, z: np.ndarray, x_p: np.ndarray) -> np.ndarray:
        """
        Generate an estimate from measurements made. Using the known locations of the landmarks, we can provide a bearing estimate.

        :param z: Measurements of the landmarks in frame, consisting of just the ECI coordinates of the landmarks with shape (N, 3)
        :param x_p: Prior state estimate consisting of position and velocity with shape (6,)
        :return: Estimate of the bearing vectors to all landmarks in the body frame with shape (N, 3)
        """
        estimate = jnp.zeros((len(z), 3))

        for i, landmark in enumerate(z):
            vec = landmark - x_p[:3]
            vec /= jnp.linalg.norm(vec)
            body_R_eci = Rotation.from_quat(self.att_quat, scalar_first=True).as_matrix()
            body_vec = body_R_eci @ vec
            estimate[i, :] = body_vec
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

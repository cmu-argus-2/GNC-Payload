from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dynamics.orbital_dynamics import f, f_jac
from vision_inference.landmark_bearings import LandmarkBearingSensor

class EKF:
    """
    Extended Kalman Filter
    """

    def __init__(self, 
                 x: np.ndarray, 
                 P: np.ndarray,
                 Q: np.ndarray, 
                 R: np.ndarray
                 ) -> None:
        """
        Initialize the EKF
        :param x: initial state consisting of position and velocity
        :param P: initial covariance
        :param Q: process noise covariance
        :param R: measurement noise covariance
        """
        self.x_m = x
        self.x_p = x
        self.P_m = P
        self.P_p = P
        self.Q = Q
        self.R = R

        config = load_config()

        # set up landmark bearing sensor and orbit determination objects
        self.landmark_bearing_sensor = LandmarkBearingSensor(config)
        # TODO: Get the actual images into the EKF. 


    def predict(self) -> None:
        """
        Predict the next prior state. This corresponds to the prior update step in the EKF algorithm.
        """
        A = f_jac(self.x_m)
        self.x_p = A @ self.x_m
        self.P_p = A @ self.P_m @ self.A.T + self.Q

    def measurement(self, z: Tuple[np.ndarray,np.ndarray]) -> None:
        """
        Update the state estimate based on the measurement. This corresponds to the posterior update step in the EKF algorithm.

        :param z: Measurement consisting of a tuple of the bearing unit vectors in the body frame and the landmark positions in ECI coordinates.
        """

        h = self.h(z[1], self.x_p)
        H = self.H(z[1], self.x_p)

        S = H @ self.P_p @ H.T + self.R
        K = self.P_p @ H.T @ np.linalg.inv(S)
        self.x_m = self.x_p + K @ (z - h)
        self.P_m = (np.eye(self.P_m.shape[0]) - K @ H) @ self.P_p


    def H(self, z: np.ndarray, x_p: np.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian of the measurement model with respect to the state.

        :param z: Measurement consisting of the landmark locations in ECI coordinates.
        :param x_p: Prior state estimate consisting of position and velocity.
        :return: The Jacobian of the measurement model with respect to the state.
        """

        jac = jax.jacobian(self.h, argnums=1)(z, x_p)
        return jac

    @staticmethod
    def h(z: np.ndarray, x_p: np.ndarray) -> np.ndarray:
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
            """
            # TODO: Implement this function. If we store our quaternion attitude as a state variable, 
            # we can use that to calculate the rotation matrix that we need to rotate the vector from ECI to body frame.
            """
            R_eci_to_body = jnp.eye(3)
            body_vec = R_eci_to_body @ vec
            estimate[i,:] = body_vec
        return estimate

def load_config() -> dict[str, Any]:
    """
    Load the configuration file and modify it for the purposes of this test.

    :return: The modified configuration file as a dictionary.
    """
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # decrease world update rate since we only care about position dynamics
    config["solver"]["world_update_rate"] = 1 / 60  # Hz
    config["mission"]["duration"] = 3 * 90 * 60  # s, roughly 1 orbit

    return config


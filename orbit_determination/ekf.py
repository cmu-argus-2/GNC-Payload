""" Extended Kalman Filter for orbit determination """

from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation
import yaml

from dynamics.orbital_dynamics import f_jac
from sensors.imu import IMU
from vision_inference.landmark_bearings import LandmarkBearingSensor


class EKF:
    """
    Extended Kalman Filter
    """

    def __init__(
        self,
        r: np.ndarray,
        v: np.ndarray,
        q: np.quaternion,
        a_b: np.ndarray,
        w_b: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        dt: float
    ) -> None:
        """
        Initialize the EKF

        :param r: Initial position state expressed in inertial frame (ECEF) with shape (3,)
        :param v: Initial velocity state expressed in body frame with shape (3,)
        :param q: Initial attitude quaternion with shape (4,) where the scalar component is first.
            Note: The quaternion is of type numpy.quaternion, not np.ndarray and assumed to be normalized
        :param a_b: Initial accelerometer bias with shape (3,)
        :param w_b: Initial angular velocity bias with shape (3,)
        :param P: Initial covariance with shape (16, 16)
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

        self.a_b = a_b
        self.w_b = w_b

        self.P_m = P
        self.P_p = P

        self.Q = Q
        self.R = R
        self.dt = dt

        self.h = np.array([[0,0,0][1,0,0][0,1,0][0,0,1]]) # Helper mapping from R3 to R4 with 0 scalar
        self.t = np.diag([1,-1,-1,-1]) 

        self.cond_threshold = 1e15
        self.g = 9.80665 # m/s^2, standard gravity Could also use -G*M*r/r^2

        config = load_config()

        # set up landmark bearing sensor and orbit determination objects
        self.landmark_bearing_sensor = LandmarkBearingSensor(config)
        # TODO: Get the actual images into the EKF.

    def Drptoq(self, rp) -> np.ndarray:
        """
        Compute the jacobian of Rodrigues parameters to quaternion
        """
        return (1/np.sqrt(1 + np.dot(rp, rp))) * self.h.T - np.outer(((1 + np.dot(rp, rp))**(-3/2)) * np.array([1, rp[0], rp[1], rp[2]]),rp)


    def predict(self, u) -> None:
        """
        Predict the next prior state. This corresponds to the prior update step in the EKF algorithm.
        Using Zac Manchester's formulation as defined in his inertial filter examples notebook 
        https://github.com/RoboticExplorationLab/inertial-filter-examples
        """

        Qm = quaternion.as_rotation_matrix(self.q_m)
        grav = np.array([0, 0, -self.g])

        af = u[:3] # accelerometer measurement from IMU
        wf = u[3:6] # angular velocity measurement from IMU

        A = f_jac(self.x_m, self.dt)
        y = quaternion.from_rotation_vector(-0.5 * self.dt * (wf - self.w_b))
        Y = quaternion.as_rotation_matrix(y)

        self.r_p = self.r_m + self.dt * Qm @ (self.v_m)
        self.q_p = self.q_m * quaternion.from_rotation_vector(0.5 * self.dt * (wf - self.w_b))

        vpk = (self.v_m + self.dt * (af - self.a_b - Qm.T @ grav)) # New velocity in old body frame
        self.v_p = Y @ vpk

        # Jacobian of the state transition function
        dvdq = self.h.T @ (self.q_m @ quaternion.as_quat_array(self.h@self.v_m) @ self.t + self.q_m.T @ quaternion.as_quat_array(self.h@self.v_m)) @ self.q_m @ self.h # This is the derivative of Q(q) w.r.t. q
        dgdq = self.h.T @ (self.q_m @ quaternion.as_quat_array(self.h@grav) + self.q_m @ quaternion.as_quat_array(self.h@grav) @ self.t) @ self.q_m # This is the derivative of Q(q)' *g w.r.t. q
        dvdb = 0.5 * self.dt * self.h.T @ y @ quaternion.as_quat_array(self.h @ vpk) + y.T @ quaternion.as_quat_array(self.h @ vpk) @ self.Drptoq(0.5 * self.dt * (wf - self.w_b))

        A = np.block([
            [np.eye(3), self.dt * dvdq, self.dt * self.Qm, np.zeros((3,6))],
            [np.zeros((3,3)), Y, np.zeros((3,6)), -0.5 * self.dt * (self.q_p @ self.h).T @ self.q_m @ self.Drptoq(0.5 * self.dt * (wf - self.w_b))],
            [np.zeros((3,3)), -self.dt @ Y @ dgdq, Y, -self.dt * Y, dvdb],
            [np.zeros((6,9)), np.eye(6)]
        ])

        self.P_p = A @ self.P_m @ A.T + self.Q


    def measurement(self, z: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Update the state estimate based on the measurement. This corresponds to the posterior update step in the EKF algorithm.

        :param z: Measurement consisting of a tuple of the bearing unit vectors in the body frame and the landmark positions in ECI coordinates with shape (N, 3)
        """

        h = self.h(z[1], self.x_p)
        H = self.H(z[1], self.x_p)

        z = z[1].reshape(-1) # Flatten the measurement vector

        #TODO: Fix self.R to have the correct dimensions based on the number of landmarks observed

        S = H @ self.P_p @ H.T + self.R
        cond = np.linalg.cond(S)

        # Check for ill-conditioned matrix and add regularization if necessary
        if cond > self.cond_threshold:
            S += np.eye(S.shape[0]) * 1e-6

        K = self.P_p @ H.T @ np.linalg.inv(S)

        delta = K @ (z - h)

        self.x_m = self.x_p + delta[0:3]
        self.q_m = self.q_p * quaternion.from_rotation_vector(delta[3:6])
        self.v_m = self.v_p + delta[6:9]
        self.a_b = self.a_b + delta[9:12]
        self.w_b = self.w_b + delta[12:15]

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
        
        :return: Estimate of the bearing vectors to all landmarks in the body frame with shape (N * 3, )
        """
        estimate = jnp.zeros((len(z) * 3))

        for i, landmark in enumerate(z):
            vec = landmark - x_p[:3]
            vec /= jnp.linalg.norm(vec)
            body_R_eci = Rotation.from_quat(self.q, scalar_first=True).as_matrix()
            body_vec = body_R_eci @ vec
            estimate = estimate.at[i*3:i*3+3].set(body_vec)
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

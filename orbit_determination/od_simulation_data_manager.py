"""
Module to manage simulation data for orbit determination.
"""

# pylint: disable=import-error
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from brahe.epoch import Epoch

from orbit_determination.landmark_bearing_sensors import LandmarkBearingSensor
from utils.brahe_utils import increment_epoch
from sensors.camera_model import CameraModel


@dataclass
class ODSimulationDataManager:
    """
    A class to store data for orbit determination simulations.
    This class also handles incrementally updating the data as new states and measurements are added.

    :param starting_epoch: The epoch at which the simulation starts.
    :param dt: The time step for the simulation.
    :param states: A numpy array of shape (N, 6) containing
                   the positions and velocities of the satellite at each time step.
    :param eci_Rs_body: A numpy array of shape (N, 3, 3) containing
                           the rotation matrices from the body frame to ECI at each time step.
    :param measurement_indices: A numpy array of shape (M,) containing the indices at which measurements were taken.
                                This is guaranteed to be non-strictly increasing, since it will contain duplicates if
                                multiple measurements were taken at the same time.
    :param bearing_unit_vectors: A numpy array of shape (M, 3) containing unit vectors towards the landmarks
                                 in the body frame.
    :param landmarks: A numpy array of shape (M, 3) containing the positions of the landmarks in ECI.
    """

    starting_epoch: Epoch
    dt: float

    states: np.ndarray = field(default_factory=lambda: np.zeros(shape=(0, 6)))
    eci_Rs_body: np.ndarray = field(default_factory=lambda: np.zeros(shape=(0, 3, 3)))

    measurement_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    measurement_camera_names: np.ndarray = field(default_factory=lambda: np.array([], dtype=str))
    bearing_unit_vectors: np.ndarray = field(default_factory=lambda: np.zeros(shape=(0, 3)))
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros(shape=(0, 3)))

    @property
    def state_count(self) -> int:
        """
        :return: The number of states in the simulation data.
        """
        return self.states.shape[0]

    @property
    def measurement_count(self) -> int:
        """
        :return: The number of measurements in the simulation data.
        """
        return len(self.measurement_indices)

    @property
    def latest_epoch(self) -> Epoch:
        """
        :return: The epoch at which the latest state was recorded.
        """
        return increment_epoch(self.starting_epoch, (self.state_count - 1) * self.dt)

    @property
    def latest_state(self) -> np.ndarray:
        """
        :return: The latest state in the simulation data.
        """
        return self.states[-1, :]

    @property
    def latest_attitude(self) -> np.ndarray:
        """
        :return: The latest attitude in the simulation data, as a rotation matrix from the body frame to ECI.
        """
        return self.eci_Rs_body[-1, ...]

    @property
    def latest_measurements(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: A tuple containing the bearing unit vectors and landmarks for the latest measurements.
        """
        indices = self.measurement_indices == self.state_count - 1
        return self.bearing_unit_vectors[indices, :], self.landmarks[indices, :]

    def assert_invariants(self) -> None:
        """
        Validates the invariants of the simulation data.

        :raises AssertionError: If any of the invariants are violated.
        """
        assert len(self.states.shape) == 2, "States must be a 2D array"
        assert self.states.shape[1] == 6, "States must have shape (N, 6)"
        assert len(self.eci_Rs_body.shape) == 3, "eci_Rs_body must be a 3D array"
        assert self.eci_Rs_body.shape[1:] == (3, 3), "eci_Rs_body must have shape (N, 3, 3)"
        assert (
            self.states.shape[0] == self.eci_Rs_body.shape[0]
        ), "states and eci_Rs_body must have the same number of entries"

        assert len(self.measurement_indices.shape) == 1, "measurement_indices must be a 1D array"
        assert (
            len(self.measurement_camera_names.shape) == 1
        ), "measurement_camera_names must be a 1D array"
        assert len(self.bearing_unit_vectors.shape) == 2, "bearing_unit_vectors must be a 2D array"
        assert (
            self.bearing_unit_vectors.shape[1] == 3
        ), "bearing_unit_vectors must have shape (M, 3)"
        assert len(self.landmarks.shape) == 2, "landmarks must be a 2D array"
        assert self.landmarks.shape[1] == 3, "landmarks must have shape (M, 3)"
        assert (
            self.measurement_indices.shape[0]
            == self.measurement_camera_names.shape[0]
            == self.bearing_unit_vectors.shape[0]
            == self.landmarks.shape[0]
        ), "measurement_indices, measurement_camera_names, bearing_unit_vectors, and landmarks must have the same number of entries"

        assert np.all(self.measurement_indices >= 0), "measurement_indices must be non-negative"
        assert np.all(
            np.diff(self.measurement_indices) >= 0
        ), "measurement_indices must be non-strictly increasing"

    def push_next_state(self, state: np.ndarray, eci_R_body: np.ndarray) -> None:
        """
        Append a new state to the simulation data.

        Args:
            state: A numpy array of shape (6,) containing the position and velocity of the satellite.
            eci_R_body: A numpy array of shape (3, 3) containing the rotation matrix from the body frame to ECI.
        """
        self.states = np.row_stack((self.states, state))
        self.eci_Rs_body = np.concatenate((self.eci_Rs_body, eci_R_body[np.newaxis, ...]), axis=0)

        self.assert_invariants()

    def take_measurement(
        self, landmark_bearing_sensor: LandmarkBearingSensor, camera_model: CameraModel
    ) -> None:
        """
        Take a measurement at the latest state and append it to the simulation data.

        :param landmark_bearing_sensor: The landmark bearing sensor to use to take the measurement.
        :param camera_model: The camera model to use to take the measurement.
        """
        t_idx = self.state_count - 1

        position_eci = self.states[t_idx, :3]
        eci_R_body = self.eci_Rs_body[t_idx, ...]

        bearing_unit_vectors, landmarks = landmark_bearing_sensor.take_measurement(
            self.latest_epoch, position_eci, eci_R_body, camera_model
        )
        measurement_count = bearing_unit_vectors.shape[0]
        assert landmarks.shape[0] == measurement_count

        self.measurement_indices = np.concatenate(
            (self.measurement_indices, np.repeat(t_idx, measurement_count))
        )
        self.measurement_camera_names = np.concatenate(
            (self.measurement_camera_names, np.repeat(camera_model.camera_name, measurement_count))
        )
        self.bearing_unit_vectors = np.concatenate(
            (self.bearing_unit_vectors, bearing_unit_vectors), axis=0
        )
        self.landmarks = np.concatenate((self.landmarks, landmarks), axis=0)

        self.assert_invariants()

"""
Module to solve the orbit determination problem using non-linear least squares.
"""

from collections.abc import Callable

import numpy as np
from brahe.constants import GM_EARTH, R_EARTH
from scipy.optimize import least_squares
from scipy.stats import circmean, circvar

# pylint: disable=import-error
from dynamics.orbital_dynamics import f, f_jac
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager


class OrbitDetermination:
    """
    A class for solving the orbit determination problem using non-linear least squares.
    """

    def __init__(self, dt: float) -> None:
        """
        Initialize the OrbitDetermination object.

        :param dt: The amount of time between each time step.
        """
        self.dt = dt

    # pylint: disable=too-many-locals
    # pylint: disable=unreachable
    def fit_circular_orbit(
        self, measurement_indices: np.ndarray, positions: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Fits a circular orbit model to a set of timestamped ECI position estimates.
        This is used for creating an initial guess for the non-linear least squares problem.

        :param measurement_indices: The measurement indices as a numpy array of shape (n,).
        :param positions: The position estimates in ECI as a numpy array of shape (n, 3).
        :return: A function that maps indices to position estimates based on the resulting circular orbit model.
        """
        assert len(positions.shape) == 2, "positions must be a 2D array"
        assert positions.shape[1] == 3, "positions must have 3 columns"
        assert positions.shape[0] >= 3, "There must be at least 3 points"
        assert len(measurement_indices.shape) == 1, "measurement_indices must be a 1D array"
        assert (
            len(measurement_indices) == positions.shape[0]
        ), "measurement_indices and positions must have the same length"
        # pylint: disable=pointless-string-statement
        """
        We want to solve for the unit normal vector of the best fit plane that passes through the origin. 
        This means we want to minimize np.linalg.norm(positions @ normal) subject to np.linalg.norm(normal) == 1.
        This is solved by computing the right singular vector corresponding to the smallest singular value of positions.
        """
        *_, vt = np.linalg.svd(positions)
        normal = vt[-1, :]

        projected_positions = positions - np.outer(positions @ normal, normal)
        orbital_radius = np.mean(np.linalg.norm(projected_positions, axis=1))
        speed = np.sqrt(GM_EARTH / orbital_radius)
        angular_speed = speed / orbital_radius

        # choose basis vectors for the orbital plane
        # there are 2 cases to avoid scenarios where the cross product is close to zero
        if abs(normal[0]) <= abs(normal[1]):
            # normal is closer to the y-axis, so we will use the x-axis in the cross product
            y_axis = np.cross(normal, np.array([1, 0, 0]))
            y_axis = y_axis / np.linalg.norm(y_axis)
            if y_axis[1] < 0:
                y_axis = -y_axis
            x_axis = np.cross(y_axis, normal)
        else:
            # normal is closer to the x-axis, so we will use the y-axis in the cross product
            x_axis = np.cross(np.array([0, 1, 0]), normal)
            x_axis = x_axis / np.linalg.norm(x_axis)
            if x_axis[0] < 0:
                x_axis = -x_axis
            y_axis = np.cross(normal, x_axis)

        positions_2d = projected_positions @ np.column_stack((x_axis, y_axis))
        angles = np.arctan2(positions_2d[:, 1], positions_2d[:, 0])

        # case 1: the orbit is counter-clockwise in the chosen basis
        phases_ccw = angles - angular_speed * measurement_indices * self.dt

        # case 2: the orbit is clockwise in the chosen basis
        phases_cw = angles + angular_speed * measurement_indices * self.dt

        # choose the case with the smaller variance
        if circvar(phases_ccw) < circvar(phases_cw):
            angular_velocity = angular_speed
            angle_0 = circmean(phases_ccw)
        else:
            angular_velocity = -angular_speed
            angle_0 = circmean(phases_cw)

        def model(ts: np.ndarray) -> np.ndarray:
            """
            Maps timestamps to state estimates.

            :param ts: The timestamps to map to state estimates as a numpy array of shape (m,).
            :return: The resulting state estimates as a numpy array of shape (m, 6).
            """
            angles_ = angular_velocity * ts * self.dt + angle_0
            positions_2d_ = orbital_radius * np.column_stack((np.cos(angles_), np.sin(angles_)))
            positions_ = positions_2d_ @ np.row_stack((x_axis, y_axis))

            velocity_directions_ = np.sign(angular_velocity) * np.cross(normal, positions_)
            velocity_directions_ = velocity_directions_ / np.linalg.norm(
                velocity_directions_, axis=1, keepdims=True
            )
            velocities_ = speed * velocity_directions_

            return np.column_stack((positions_, velocities_))

        return model

    def fit_orbit(
        self, data_manager: ODSimulationDataManager, semi_major_axis_guess: float = R_EARTH + 600e3
    ) -> np.ndarray:
        """
        Solve the orbit determination problem using non-linear least squares.

        :param data_manager: The ODSimulationDataManager object containing the simulation data.
        :param semi_major_axis_guess: An initial guess for the semi-major axis of the satellite's orbit.
        :return: A numpy array of shape (data_manager.state_count, 6) containing
                 the estimated ECI position and velocity of the satellite at each time step.
        """
        data_manager.assert_invariants()
        N = data_manager.state_count
        M = data_manager.measurement_count

        measurement_eci_Rs_body = data_manager.eci_Rs_body[data_manager.measurement_indices, ...]
        bearing_unit_vectors_wf = np.einsum(
            "ijk,ik->ij", measurement_eci_Rs_body, data_manager.bearing_unit_vectors
        )

        def residuals(X: np.ndarray) -> np.ndarray:
            """
            Compute the residuals of the non-linear least squares problem.

            :param X: A flattened numpy array of shape (6 * N,) containing
                      the ECI positions and velocities of the satellite at each time step.
            :return: A numpy array of shape (6 * (N - 1) + 3 * M,) containing the residuals.
            """
            states = X.reshape(N, 6)
            res = np.zeros(6 * (N - 1) + 3 * M)
            idx = 0  # index into res

            # dynamics residuals
            for i in range(N - 1):
                res[idx : idx + 6] = states[i + 1, :] - f(states[i, :], self.dt)
                idx += 6

            # measurement residuals
            for i, (time, landmark) in enumerate(
                zip(data_manager.measurement_indices, data_manager.landmarks)
            ):
                cubesat_position = states[time, :3]
                predicted_bearing = landmark - cubesat_position
                predicted_bearing_unit_vector = predicted_bearing / np.linalg.norm(
                    predicted_bearing
                )

                res[idx : idx + 3] = predicted_bearing_unit_vector - bearing_unit_vectors_wf[i]
                idx += 3

            assert idx == len(res)
            print(np.sum(res**2))
            return res

        def residual_jac(X: np.ndarray) -> np.ndarray:
            """
            Compute the Jacobian of the residuals of the non-linear least squares problem.

            :param X: A flattened numpy array of shape (6 * N,) containing
                      the ECI positions and velocities of the satellite at each time step.
            :return: A numpy array of shape (6 * (N - 1) + 3 * M, 6 * N) containing the Jacobian of the residuals.
            """
            states = X.reshape(N, 6)
            jac = np.zeros((6 * (N - 1) + 3 * M, 6 * N), dtype=X.dtype)
            row_idx = 0  # row index into jac
            # Note that indices into the columns of jac are 6 * i : 6 * (i + 1) for the ith state

            # dynamics Jacobian
            for i in range(N - 1):
                jac[row_idx : row_idx + 6, row_idx : row_idx + 6] = -f_jac(states[i, :], self.dt)
                jac[row_idx : row_idx + 6, row_idx + 6 : row_idx + 12] = np.eye(6)
                row_idx += 6

            # measurement Jacobian
            for i, (time, landmark) in enumerate(
                zip(data_manager.measurement_indices, data_manager.landmarks)
            ):
                cubesat_position = states[time, :3]
                predicted_bearing = landmark - cubesat_position
                predicted_bearing_norm = np.linalg.norm(predicted_bearing)
                predicted_bearing_unit_vector = predicted_bearing / predicted_bearing_norm

                jac[row_idx : row_idx + 3, 6 * time : 6 * time + 3] = (
                    np.outer(predicted_bearing_unit_vector, predicted_bearing_unit_vector)
                    - np.eye(3)
                ) / predicted_bearing_norm
                row_idx += 3

            assert row_idx == jac.shape[0]
            return jac

        # fit a circular orbit to the altitude-normalized landmarks to get an initial guess
        altitude_normalized_landmarks = data_manager.landmarks / np.linalg.norm(
            data_manager.landmarks, axis=1, keepdims=True
        )
        model = self.fit_circular_orbit(
            data_manager.measurement_indices, semi_major_axis_guess * altitude_normalized_landmarks
        )
        # pylint: disable=no-member
        initial_guess = model(np.arange(N)).flatten()

        result = least_squares(residuals, initial_guess, method="lm", jac=residual_jac)
        return result.x.reshape(N, 6)

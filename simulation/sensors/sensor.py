"""
Module that implements a sensor class that adds noise to a clean signal.
"""

from math import sqrt

import numpy as np
from sensors.bias import Bias, BiasParams


# pylint: disable=R0903
class SensorNoiseParams:
    """
    Sensor noise parameters.
    """

    def __init__(
        self,
        bias_params: BiasParams,
        sigma_v: float,
        scale_factor_error: float,
    ) -> None:
        """
        Parameters for a time-varying bias modeled as a random walk

        Args:
            biasParams (BiasParams): bias parameters
            sigma_w (float): continuous-time power spectral density of additive white noise
            to sensor output [units/sqrt(Hz)]
            scale_factor_error (float): multiplier [-]
        """
        self.bias_params = bias_params
        self.sigma_v = sigma_v
        self.scale_factor_error = scale_factor_error

    @staticmethod
    def get_random_params(
        bias_params: BiasParams,
        sigma_v_range: list,
        scale_factor_error_range: list,
    ) -> "SensorNoiseParams":
        """
        Getter for random bias parameters

        Args:
            biasParams (BiasParams): bias parameters
            sigma_v_range (list): [min, max]
            scale_factor_error_range (list) [min, max]

        Returns:
            SensorNoiseParams: sensor noise parameters
        """
        return SensorNoiseParams(
            bias_params,
            np.random.uniform(*sigma_v_range),
            np.random.uniform(*scale_factor_error_range),
        )


class Sensor:
    """
    Sensor class.
    """

    def __init__(self, dt: float, sensor_noise_params: SensorNoiseParams) -> None:
        """
        Sensor class that adds noise to a clean signal.

        Args:
            dt (float): The time step for the simulation.
            sensor_noise_params (SensorNoiseParams): The noise parameters for the sensor.
        """
        self.dt = dt
        self.bias = Bias(dt, sensor_noise_params.bias_params)

        # discrete version of sensor_noise_params.sigma_v causing the bias to random walk when integrated
        self.white_noise = sensor_noise_params.sigma_v / sqrt(dt)

        self.scale_factor_error = sensor_noise_params.scale_factor_error

        self.value = 0

    def update(self, clean_signal: np.ndarray) -> np.ndarray:
        """
        Update the measurements of the sensor by applying noise to the clean signal.

        Args:
            clean_signal (np.ndarray): The clean signal.

        Returns:
            np.ndarray: The noisy signal.
        """
        self.bias.update()
        noise = self.white_noise * np.random.standard_normal()
        self.value = (1 + self.scale_factor_error) * clean_signal + self.bias.get_bias() + noise
        return self.value

    def get_bias(self) -> float:
        """
        Getter for the bias
        """
        return self.bias.get_bias()

    def get_value(self) -> float:
        """
        Getter for the latest signal value
        """
        return self.value


class TriAxisSensor:
    """
    Triaxis sensor class.
    """

    def __init__(self, dt: float, axes_params: list, misalignment_range: list) -> None:
        """
        Class that creates a noisy tri-axis signal.

        Args:
            dt (float): The time step for the simulation.
            axes_params (IMUNoiseParams): The noise parameters for the sensor.
            misalignment_range (list): The misalignment range for the sensor.
        """
        self.dt = dt
        self.x = Sensor(dt, axes_params[0])
        self.y = Sensor(dt, axes_params[1])
        self.z = Sensor(dt, axes_params[2])

        self.misalignmentxy = np.random.uniform(*misalignment_range)
        self.misalignmentxz = np.random.uniform(*misalignment_range)
        self.misalignmentyz = np.random.uniform(*misalignment_range)

    def get_bias(self) -> np.ndarray:
        """
        Getter for the bias
        """
        return np.array([self.x.get_bias(), self.y.get_bias(), self.z.get_bias()])

    def update(self, clean_signal: np.ndarray) -> np.ndarray:
        """
        Update the measurements of the TriAxisSensor by applying noise to the clean signal.

        Args:
            clean_signal (np.ndarray): The clean signal.

        Returns:
            np.ndarray: The noisy signal.
        """
        self.x.update(clean_signal[0])
        self.y.update(clean_signal[1])
        self.z.update(clean_signal[2])

        # Apply misalignment to each sensor
        signal_x = (
            self.x.get_value()
            + self.misalignmentxy * clean_signal[1]
            + self.misalignmentxz * clean_signal[2]
        )
        signal_y = (
            self.y.get_value()
            + (-1) * self.misalignmentxy * clean_signal[0]
            + self.misalignmentyz * clean_signal[2]
        )
        signal_z = (
            self.z.get_value()
            + (-1) * self.misalignmentxz * clean_signal[0]
            + (-1) * self.misalignmentyz * clean_signal[1]
        )

        return np.array(
            [
                signal_x,
                signal_y,
                signal_z,
            ]
        )

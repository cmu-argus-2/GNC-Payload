"""
Module that implements a sensor class that adds noise to a clean signal.
"""
from math import sqrt

import numpy as np

from sensors.bias import Bias, BiasParams


class SensorNoiseParams:
    def __init__(self, biasParams: BiasParams, sigma_v: float, scale_factor_error: float) -> None:
        """Parameters for a time-varying bias modeled as a random walk

        Args:
            biasParams (BiasParams): bias parameters
            sigma_w (float): continuous-time power spectral density of additive white noise to sensor output [units/sqrt(Hz)]
            scale_factor_error (float): multiplier [-]
        """
        self.bias_params = biasParams
        self.sigma_v = sigma_v
        self.scale_factor_error = scale_factor_error

    @staticmethod
    def get_random_params(
        biasParams: BiasParams, sigma_v_range: list, scale_factor_error_range: list
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
            biasParams,
            np.random.uniform(*sigma_v_range),
            np.random.uniform(*scale_factor_error_range),
        )


class Sensor:
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
        return (1 + self.scale_factor_error) * clean_signal + self.bias.get_bias() + noise

    def get_bias(self) -> float:
        """
        Getter for the bias
        """
        return self.bias.get_bias()


class TriAxisSensor:
    def __init__(self, dt: float, axes_params: list) -> None:
        """
        Class that creates a noisy tri-axis signal.

        Args:
            dt (float): The time step for the simulation.
            axes_params (IMUNoiseParams): The noise parameters for the sensor.
        """
        self.dt = dt
        self.x = Sensor(dt, axes_params[0])
        self.y = Sensor(dt, axes_params[1])
        self.z = Sensor(dt, axes_params[2])

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
        return np.array(
            [
                self.x.update(clean_signal[0]),
                self.y.update(clean_signal[1]),
                self.z.update(clean_signal[2]),
            ]
        )

"""
Module implementing sensor bias
"""
from math import sqrt

import numpy as np


class BiasParams:
    def __init__(self, initial_bias: float, sigma_w: float) -> None:
        """
        Parameters for a time-varying bias modeled as a random walk

        Args:
            initial_bias (float): [units]
            sigma_w (float): continuous-time power spectral density of additive white noise 
            to time-derivative of bias. [(units/s)/sqrt(Hz)]
        """
        self.initial_bias = initial_bias
        self.sigma_w = sigma_w

    @staticmethod
    def get_random_params(
        initial_bias_range: list, sigma_w_range: list
    ) -> "BiasParams":
        """
        Getter for random bias parameters

        Args:
            initial_bias_range (list): [min, max]
            sigma_w_range (list): [min, max]

        Returns:
            BiasParams: bias parameters
        """
        return BiasParams(np.random.uniform(*initial_bias_range), np.random.uniform(*sigma_w_range))


class Bias:
    def __init__(self, dt: float, bias_params: BiasParams) -> None:
        """Initialize a time-varying bias modeled as a random walk

        Args:
            dt (float): delta time [s]
            bias_params (BiasParams): bias parameters
        """
        self.dt = dt
        self.bias = bias_params.initial_bias

        # discrete version of sigma_w causing the bias to random walk when integrated
        self.sigma_random_walk_ = bias_params.sigma_w / sqrt(dt)

    def update(self) -> float:
        """
        Update the bias
        """
        noise = self.sigma_random_walk_ * np.random.standard_normal()
        self.bias += self.dt * noise
        return self.bias

    def get_bias(self) -> float:
        """
        Getter for the bias
        """
        return self.bias

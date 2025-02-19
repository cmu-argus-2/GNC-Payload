"""
Inertial Measurement Unit (IMU) Sensor Module
"""

import numpy as np

from sensors.sensor import TriAxisSensor


class IMUNoiseParams:
    def __init__(self, gyro_params: list, accel_params: list) -> None:
        """Gyroscope and Accelerometer parameters

        Args:
            gyro_params ([SensorNoiseParams x 3]): list of SensorNoiseParams, one per x, y, z axes
            accel_params ([SensorNoiseParams x 3]): list of SensorNoiseParams, one per x, y, z axes
        """
        self.gyro = gyro_params
        self.accel = accel_params


class IMU:
    def __init__(self, dt: float, IMU_noise_params: IMUNoiseParams) -> None:
        """
        Initialize an IMU sensor with given noise parameters.

        Args:
            dt (float): The time step for the simulation.
            IMU_noise_params (IMUNoiseParams): The noise parameters for the IMU sensor.
        """
        self.gyro = TriAxisSensor(dt, IMU_noise_params.gyro)
        self.accel = TriAxisSensor(dt, IMU_noise_params.accel)

    def get_bias(self) -> tuple:
        """
        Return the bias of the IMU sensor.
        """
        gyro_bias = self.gyro.get_bias()
        accel_bias = self.accel.get_bias()
        return gyro_bias, accel_bias

    def update(self, clean_gyro_signal: np.ndarray, clean_accel_signal: np.ndarray) -> tuple:
        """
        Update the measurements of the IMU sensor by applying noise to the clean signals.

        Args:
            clean_gyro_signal (np.ndarray): The clean angular velocity signal.
            clean_accel_signal (np.ndarray): The clean acceleration signal.
        """
        gyro_measurement = self.gyro.update(clean_gyro_signal)
        accel_measurement = self.accel.update(clean_accel_signal)
        return gyro_measurement, accel_measurement

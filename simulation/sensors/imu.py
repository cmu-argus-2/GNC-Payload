"""
Inertial Measurement Unit (IMU) Sensor Module
"""

import numpy as np
from .bias import BiasParams
from .sensor import SensorNoiseParams, TriAxisSensor


# pylint: disable=R0903
class IMUNoiseParams:
    """
    IMU Noise parameters.
    """

    def __init__(self, gyro_params: list, accel_params: list) -> None:
        """Gyroscope and Accelerometer parameters

        Args:
            gyro_params ([SensorNoiseParams x 3]): list of SensorNoiseParams, one per x, y, z axes
            accel_params ([SensorNoiseParams x 3]): list of SensorNoiseParams, one per x, y, z axes
        """
        self.gyro = gyro_params
        self.accel = accel_params


class IMU:
    """
    IMU class.
    """

    def __init__(
        self, dt: float, imu_noise_params: IMUNoiseParams, misalignment_range: list
    ) -> None:
        """
        Initialize an IMU sensor with given noise parameters.

        Args:
            dt (float): The time step for the simulation.
            IMU_noise_params (IMUNoiseParams): The noise parameters for the IMU sensor.
            misalignment_range (list): The misalignment range for the sensor.
        """
        self.gyro = TriAxisSensor(dt, imu_noise_params.gyro, misalignment_range)
        self.accel = TriAxisSensor(dt, imu_noise_params.accel, misalignment_range)

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

    @staticmethod
    def get_default_imu(dt: float) -> "IMU":
        """
        Initializes the IMU.

        :param dt: The time step for the simulation.

        :return: The initialized IMU.
        """
        # Initialize the IMU
        # bias params are min max range of bias and sigma_w
        # [units] and [(units/s)/sqrt(Hz)]
        bias_params_x = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
        bias_params_y = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
        bias_params_z = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
        # bias_params = BiasParams.get_random_params([0, 0], [0, 0])
        # sigma_v [units/sqrt(Hz)] & scale_factor_error [-]
        sensor_noise_params_accel_x = SensorNoiseParams.get_random_params(
            bias_params_x, [0, 0.0], [0, 0.0]
        )
        sensor_noise_params_accel_y = SensorNoiseParams.get_random_params(
            bias_params_y, [0, 0.0], [0, 0.0]
        )
        sensor_noise_params_accel_z = SensorNoiseParams.get_random_params(
            bias_params_z, [0, 0.0], [0, 0.0]
        )
        sensor_noise_params_accel = [
            sensor_noise_params_accel_x,
            sensor_noise_params_accel_y,
            sensor_noise_params_accel_z,
        ]
        # sigma_v [units/sqrt(Hz)] & scale_factor_error [-]
        sensor_noise_params_gyro_x = SensorNoiseParams.get_random_params(
            bias_params_x, [1e-6, 1e-5], [0, 0.01]
        )
        sensor_noise_params_gyro_y = SensorNoiseParams.get_random_params(
            bias_params_y, [1e-6, 1e-5], [0, 0.01]
        )
        sensor_noise_params_gyro_z = SensorNoiseParams.get_random_params(
            bias_params_z, [1e-6, 1e-5], [0, 0.01]
        )
        sensor_noise_params_gyro = [
            sensor_noise_params_gyro_x,
            sensor_noise_params_gyro_y,
            sensor_noise_params_gyro_z,
        ]

        imu_noise_params = IMUNoiseParams(
            gyro_params=sensor_noise_params_gyro, accel_params=sensor_noise_params_accel
        )
        imu = IMU(
            dt=dt,
            imu_noise_params=imu_noise_params,
            misalignment_range=[-0.01, 0.01],
        )

        return imu

import numpy as np

from src.py.configs.filter_config import FilterConfig
from src.py.filter.complementary_filter import ComplementaryFilter
from src.py.filter.low_pass_filter import LowPassFilter
from src.py.record_data.file_manager import FileManager


class DataProcessor:
    def __init__(
            self,
            file_manager: FileManager,
            filter_config:FilterConfig,
            gyro_bias=None,
            accel_bias=None
    ):
        self.file_manager = file_manager
        self.config = filter_config
        self.gyro_bias = gyro_bias
        self.accel_bias = accel_bias
        self.low_pass_filter = LowPassFilter(filter_config.alpha_low_pass)
        self.complementary_filter = ComplementaryFilter(filter_config.alpha_complementary)

        self.R_gravity = None
        self.gravity_mag = None

    def process(self, gyro, acc, dt):
        corrected_gyro = gyro - self.gyro_bias
        corrected_acc = acc - self.accel_bias
        filtered_gyro = self.low_pass_filter.filter(corrected_gyro)
        filtered_acc = self.low_pass_filter.filter(corrected_acc)
        rotated_acc = self.R_gravity.apply(filtered_acc)

        if self.file_manager.recording:
            self.file_manager.update({
                "raw": [*acc, *gyro, dt],
                "corrected": [*corrected_acc, *corrected_gyro, dt],
                "filtered": [*filtered_acc, *filtered_gyro, dt],
                "rotated": [*rotated_acc, *filtered_gyro, dt]
            })

        return filtered_gyro, rotated_acc

    def update_bias(self, gyro_bias=None, accel_bias=None):
        if gyro_bias is None and accel_bias is None:
            print("No bias values provided.")
            return
        self.gyro_bias = gyro_bias
        self.accel_bias = accel_bias

    def is_stationary(self, accel_values, gyro_values, accel_stationary_low=0.95, accel_stationary_high=1.05,
                      gyro_stationary_threshold=2):
        """
        Checks if the sensor is stationary based on accelerometer and gyroscope readings.

        Args:
            accel_values (np.array): [x_accel, y_accel, z_accel]
            gyro_values (np.array): [x_gyro, y_gyro, z_gyro]
            accel_stationary_low (float): Lower bound for stationary acceleration magnitude.
            accel_stationary_high (float): Upper bound for stationary acceleration magnitude.
            gyro_stationary_threshold (float): Maximum magnitude for stationary gyroscope readings.

        Returns:
            bool: True if stationary, False otherwise.
        """
        corrected_gyro = gyro_values - self.gyro_bias

        filtered_gyro = self.low_pass_filter.filter(corrected_gyro)

        accel_magnitude = np.linalg.norm(accel_values)
        gyro_magnitude = np.linalg.norm(filtered_gyro)

        is_accel_stationary = accel_stationary_low < accel_magnitude < accel_stationary_high
        is_gyro_stationary = gyro_magnitude < gyro_stationary_threshold

        if not is_accel_stationary:
            print(f"Accelerometer not stationary: {accel_magnitude}")
        if not is_gyro_stationary:
            print(f"Gyroscope not stationary: {gyro_magnitude}")

        return is_accel_stationary and is_gyro_stationary
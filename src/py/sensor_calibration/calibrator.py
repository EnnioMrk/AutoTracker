from typing import Any

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation

from src.py.record_data.serial_reader import SerialReader
from src.py.data_analysis.data_processor import DataProcessor
from src.py.record_data.record_data import data_processor


class SensorCalibrator:
    def __init__(self, config, ser: SerialReader, data_processor: DataProcessor):
        self.ser = ser
        self.data_processor = data_processor
        self.config = config
        self.iterations = config.iterations

    def setup_stationary(self):
        data = self.calibrate_stationary()
        if data is None:
            return False
        accel, gyro, accel_bias, gyro_bias = data

        self.data_processor.update_bias(gyro_bias, accel_bias)
        R_calib, gravity_mag = self.compute_gravity_alignment(accel)

        data_processor.R_gravity = R_calib
        data_processor.gravity_mag = gravity_mag

        return True

    def calibrate_stationary(self) -> tuple[list[list[Any]], list[list[Any]], Any, Any] | None:
        iteration = 0
        not_stationary = 0

        a = []
        g = []

        while iteration < self.iterations:
            if not_stationary > 100:
                print("Sensor is not stationary. Please stop moving.")
                return None
            data = self.ser.read_data()
            if data is None:
                continue
            ax, ay, az, gx, gy, gz, dt = data

            stationary = self.data_processor.is_stationary(
                [ax, ay, az], [gx, gy, gz]
            )

            if not stationary:
                not_stationary += 1
                continue

            a.append([ax, ay, az])
            g.append([gx, gy, gz])

            iteration += 1

        accel_bias = np.mean(a, axis=0)
        gyro_bias = np.mean(g, axis=0)

        return a, g, accel_bias, gyro_bias

    @staticmethod
    def compute_mean_acceleration(measurements: list[list[Any]], trim_percentage: float = 0.1):
        """
            Compute the average raw acceleration vector and its magnitude using trimmed mean.
            """
        if not 0.0 <= trim_percentage <= 0.5:
            raise ValueError("trim_percentage must be between 0.0 and 0.5")

        # Convert measurements to a numpy array
        vectors = np.array([m for m in measurements])
        num_measurements = vectors.shape[0]

        if num_measurements == 0:
            return np.zeros(3, dtype=np.float64), 0.0

        # Create trimmed vectors by component
        trimmed_x = np.sort(vectors[:, 0])[
                    int(num_measurements * trim_percentage):int(num_measurements * (1 - trim_percentage))]
        trimmed_y = np.sort(vectors[:, 1])[
                    int(num_measurements * trim_percentage):int(num_measurements * (1 - trim_percentage))]
        trimmed_z = np.sort(vectors[:, 2])[
                    int(num_measurements * trim_percentage):int(num_measurements * (1 - trim_percentage))]

        # Calculate means
        mean_vec = np.array([np.mean(trimmed_x), np.mean(trimmed_y), np.mean(trimmed_z)])

        gravity_mag = np.linalg.norm(mean_vec)
        if gravity_mag > 0:
            g_normalized = mean_vec / gravity_mag
        else:
            g_normalized = np.array([0, 0, 1])  # Default to world up if no signal

        return g_normalized, gravity_mag

    @staticmethod
    def compute_rotation_matrix(g_normalized: ndarray) -> ndarray:
        """
            Compute rotation matrix to align measured gravity with world up [0, 0, 1].
            """
        world_up = np.array([0, 0, 1], dtype=np.float64)
        v = np.cross(g_normalized, world_up)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            return np.identity(3)
        v = v / v_norm
        dot_val = np.dot(g_normalized, world_up)
        theta = np.arccos(np.clip(dot_val, -1.0, 1.0))
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        R = np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R

    def compute_gravity_alignment(self, measurements: list[list[Any]]):
        """
        Compute rotation matrix that should align with gravity using a set of raw accelerometer measurements.
        """
        g_normalized, gravity_mag = self.compute_mean_acceleration(measurements)
        R_matrix = self.compute_rotation_matrix(g_normalized)
        R_calib = Rotation.from_matrix(matrix=R_matrix)

        return R_calib, gravity_mag
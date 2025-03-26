from typing import Any

import numpy as np
from numpy import ndarray, dtype

from src.py.record_data.serial_reader import SerialReader


def get_calibration_data(ser: SerialReader, iterations: int) -> tuple[
    list[ndarray[tuple[int, ...], dtype[Any]]], list[ndarray[tuple[int, ...], dtype[Any]]], Any, Any]:
    """Reads and returns calibration data from the sensor."""
    index = 0
    accel = []
    gyro = []
    print("Collecting calibration data...")
    while index < iterations:
        data = ser.read_data()
        if data is None:
            continue

        x_accel, y_accel, z_accel, x_gyro, y_gyro, z_gyro, _ = data
        accel.append(np.array([x_accel, y_accel, z_accel]))
        gyro.append(np.array([x_gyro, y_gyro, z_gyro]))
        if index % (iterations // 10 + 1) == 0: # Feedback on progress
            print(f"Collected {index}/{iterations} measurements")
        index += 1
    print("Calibration data collection complete.")

    accel_bias = np.mean(accel, axis=0)
    gyro_bias = np.mean(gyro, axis=0)

    return accel, gyro, accel_bias, gyro_bias
from typing import Any, List

import numpy as np
from numpy import ndarray, dtype


def get_calibration_data(ser: Any, iterations: int) -> tuple[
    list[ndarray[tuple[int, ...], dtype[Any]]], list[ndarray[tuple[int, ...], dtype[Any]]], Any, Any]:
    """Reads and returns calibration data from the sensor."""
    index = 0
    accel = []
    gyro = []
    print("Collecting calibration data...")
    while index < iterations:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            values = line.split('\t')
            if len(values) >= 3: # Changed to 3, assuming you are at least getting accel data (x, y, z)
                try:
                    x_accel, y_accel, z_accel = map(float, values[:3]) # Assuming first 3 are accel data
                    x_gyro, y_gyro, z_gyro = map(float, values[3:6]) # Assuming next 3 are gyro data
                except ValueError as e: # More specific exception
                    print(f"Error converting sensor data to float: {e}, line: '{line}'")
                    continue
                except Exception as e: # Catch other potential errors
                    print(f"Unexpected error reading sensor data: {e}, line: '{line}'")
                    continue

                index += 1
                accel.append(np.array([x_accel, y_accel, z_accel]))
                gyro.append(np.array([x_gyro, y_gyro, z_gyro]))
                if index % (iterations // 10 + 1) == 0: # Feedback on progress
                    print(f"Collected {index}/{iterations} measurements")
    print("Calibration data collection complete.")

    accel_bias = np.mean(accel, axis=0)
    gyro_bias = np.mean(gyro, axis=0)

    return accel, gyro, accel_bias, gyro_bias
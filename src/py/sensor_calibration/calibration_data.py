from typing import Any, List

from src.py.sensor_calibration.calibrate import VectorMeasurement


def get_calibration_data(ser: Any, iterations: int) -> List[VectorMeasurement]:
    """Reads and returns calibration data from the sensor."""
    index = 0
    measurements = []
    print("Collecting calibration data...")
    while index < iterations:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            values = line.split('\t')
            if len(values) >= 3: # Changed to 3, assuming you are at least getting accel data (x, y, z)
                try:
                    x_accel, y_accel, z_accel = map(float, values[:3]) # Assuming first 3 are accel data
                except ValueError as e: # More specific exception
                    print(f"Error converting sensor data to float: {e}, line: '{line}'")
                    continue
                except Exception as e: # Catch other potential errors
                    print(f"Unexpected error reading sensor data: {e}, line: '{line}'")
                    continue

                index += 1
                measurements.append(VectorMeasurement(x_accel, y_accel, z_accel))
                if index % (iterations // 10 + 1) == 0: # Feedback on progress
                    print(f"Collected {index}/{iterations} measurements")
    print("Calibration data collection complete.")
    return measurements
import numpy as np
import serial
import time
from typing import List, Tuple, Any

from src.py.sensor_calibration.calibrate import VectorMeasurement, calculate_avg_accel, calibrate_sensors


def record_sensor_data_for_bias(ser: Any, duration_seconds: int = 300) -> Tuple[List[VectorMeasurement], List[VectorMeasurement]]:
    """
    Records accelerometer and gyroscope data for bias calculation over a specified duration.
    """
    print(f"Recording sensor data for bias calculation for {duration_seconds} seconds...")
    accel_measurements = []
    gyro_measurements = []
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            values = line.split('\t')
            if len(values) >= 6: # Expecting at least 6 values: ax, ay, az, gx, gy, gz
                try:
                    ax, ay, az, gx, gy, gz = map(float, values[:6])
                    accel_measurements.append(VectorMeasurement(ax, ay, az))
                    gyro_measurements.append(VectorMeasurement(gx, gy, gz))
                except ValueError as e:
                    print(f"Error converting sensor data to float: {e}, line: '{line}'")
                    continue
                except Exception as e:
                    print(f"Unexpected error reading sensor data: {e}, line: '{line}'")
                    continue
    print("Finished recording sensor data.")
    return accel_measurements, gyro_measurements


def calculate_biases(accel_measurements: List[VectorMeasurement], gyro_measurements: List[VectorMeasurement], rotation_matrix: np.ndarray, gravity_magnitude: float, trim_percentage: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates accelerometer and gyroscope biases.
    """
    if not accel_measurements or not gyro_measurements:
        print("Warning: No measurements provided for bias calculation.")
        return np.zeros(3), np.zeros(3)

    # 1. Rotate Accelerometer Data to Earth Frame
    rotated_accel_vectors = np.array([rotation_matrix @ m.vec for m in accel_measurements])
    rotated_accel_measurements = [VectorMeasurement(*vec) for vec in rotated_accel_vectors]

    # 2. Calculate Trimmed Mean of Rotated Accelerometer Data
    avg_rotated_accel_vec, _ = calculate_avg_accel(rotated_accel_measurements, trim_percentage=trim_percentage)

    # 3. Calculate Accelerometer Bias
    # Expected values in Earth frame: [0, 0, gravity_magnitude] (assuming Z-up world frame)
    expected_accel_earth_frame = np.array([0.0, 0.0, gravity_magnitude])
    accel_bias = avg_rotated_accel_vec - expected_accel_earth_frame

    # 4. Calculate Trimmed Mean of Gyroscope Data (in sensor frame, bias is just offset from zero)
    avg_gyro_vec_sensor_frame, _ = calculate_avg_accel(gyro_measurements, trim_percentage=trim_percentage) # Reusing calculate_avg_accel for trimmed mean
    gyro_bias = avg_gyro_vec_sensor_frame - np.array([0.0, 0.0, 0.0]) # Expected gyro reading is [0, 0, 0]

    return accel_bias, gyro_bias


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


if __name__ == '__main__':
    # Serial setup (replace 'COM3' with your port)
    try:
        ser = serial.Serial('COM3', 115200, timeout=1)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        exit()

    calibration_iterations = 10000
    trim_percentage_calibration = 0.1

    # 1. Sensor Calibration to get Rotation Matrix and Gravity Magnitude
    print("Starting Sensor Calibration...")
    calibration_measurements = get_calibration_data(ser, calibration_iterations)
    if not calibration_measurements:
        print("Calibration measurements failed. Exiting.")
        ser.close()
        exit()
    R, gravity_mag = calibrate_sensors(calibration_measurements, trim_percentage=trim_percentage_calibration)
    print("\nCalibration Rotation Matrix (R):\n", R)
    print("Calibration Gravity Magnitude:", gravity_mag)

    # 2. Record Data for Bias Calculation (e.g., 5 minutes = 300 seconds)
    bias_recording_duration_seconds = 600
    trim_percentage_bias = 0.1
    accel_bias_measurements, gyro_bias_measurements = record_sensor_data_for_bias(ser, duration_seconds=bias_recording_duration_seconds)

    # 3. Calculate Biases
    accel_bias, gyro_bias = calculate_biases(accel_bias_measurements, gyro_bias_measurements, R, gravity_mag, trim_percentage=trim_percentage_bias)

    print("\nBias Calculation Results:")
    print("Accelerometer Bias (x, y, z):", accel_bias)
    print("Gyroscope Bias (x, y, z):", gyro_bias)
    print("Units: Accelerometer bias in same units as raw readings (e.g., g or m/s^2), Gyro bias in same units as raw readings (e.g., deg/s or rad/s)")

    if ser.is_open:
        ser.close()
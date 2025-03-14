from typing import Any, List

import numpy as np
import serial
import time
from scipy.spatial.transform import Rotation

from src.py.sensor_calibration.ekf import EKF


def get_calibration_data(ser: Any, iterations: int) -> tuple[List[np.ndarray], List[np.ndarray], Any, Any]:
    """Reads and returns calibration data from the sensor."""
    index = 0
    accel = []
    gyro = []
    print("Collecting calibration data...")
    while index < iterations:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            values = line.split('\t')
            if len(values) >= 6:
                try:
                    x_accel, y_accel, z_accel = map(float, values[:3])
                    x_gyro, y_gyro, z_gyro = map(float, values[3:6])
                except ValueError as e:
                    print(f"Error converting sensor data to float: {e}, line: '{line}'")
                    continue
                except Exception as e:
                    print(f"Unexpected error reading sensor data: {e}, line: '{line}'")
                    continue

                index += 1
                accel.append(np.array([x_accel, y_accel, z_accel]))
                gyro.append(np.array([x_gyro, y_gyro, z_gyro]))
                if index % (iterations // 10 + 1) == 0:
                    print(f"Collected {index}/{iterations} measurements")
    print("Calibration data collection complete.")

    accel_bias = np.mean(accel, axis=0)
    gyro_bias = np.mean(gyro, axis=0)  # in °/s
    return accel, gyro, accel_bias, gyro_bias


def live_ekf_test():
    """
    Continuously collects sensor data from a serial port and compares:
      1. The EKF's orientation (Euler angles) and linear acceleration estimates.
      2. A direct integration method that integrates gyro data and subtracts gravity.
    """
    ser = serial.Serial('COM3', 115200, timeout=1)  # Adjust port and baud rate as needed
    time.sleep(2)

    calib_iterations = 100
    _, _, accel_bias, gyro_bias = get_calibration_data(ser, calib_iterations)
    # Convert gyro bias from °/s to rad/s for consistency.
    gyro_bias = np.deg2rad(gyro_bias)
    print("Calibration complete.")
    print("Accel Bias (g):", accel_bias)
    print("Gyro Bias (rad/s):", gyro_bias)

    dt_est = 0.007
    ekf = EKF(dt=dt_est, process_noise=0.01, accel_noise=0.2, gyro_noise=0.05)
    ekf.set_gravity_magnitude(9.81)

    # Initialize direct integration method with identity rotation
    raw_rotation = Rotation.identity()

    count = 0
    sum_sq_euler_diff = np.zeros(3)
    sum_sq_lin_acc_diff = np.zeros(3)

    print("Starting live sensor data collection. Press Ctrl+C to stop.")
    prev_time = time.time()

    try:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                values = line.split('\t')
                if len(values) >= 6:
                    try:
                        x_accel, y_accel, z_accel = map(float, values[:3])
                        x_gyro, y_gyro, z_gyro = map(float, values[3:6])
                    except Exception as e:
                        print(f"Error parsing sensor data: {e}, line: '{line}'")
                        continue

                    # Accelerometer: sensor outputs in g -> convert to m/s² after bias removal.
                    accel_meas = np.array([x_accel, y_accel, z_accel]) - accel_bias
                    accel_meas_ms2 = accel_meas * 9.81

                    # Gyroscope: sensor outputs in °/s -> convert to rad/s.
                    raw_gyro = np.array([x_gyro, y_gyro, z_gyro])
                    gyro_meas_rad = np.deg2rad(raw_gyro) - gyro_bias

                    current_time = time.time()
                    dt = current_time - prev_time
                    if dt <= 0:
                        dt = dt_est
                    prev_time = current_time

                    # --- EKF Pipeline ---
                    ekf.update(accel_meas_ms2, gyro_meas_rad, dt)
                    ekf_euler = ekf.get_euler_angles()  # radians
                    ekf_lin_acc = ekf.get_linear_accel(accel_meas_ms2)

                    # --- Direct Integration Pipeline ---
                    delta_rot = Rotation.from_rotvec(gyro_meas_rad * dt)
                    raw_rotation = raw_rotation * delta_rot
                    raw_euler = raw_rotation.as_euler('xyz')
                    R_raw = raw_rotation.as_matrix()
                    gravity_sensor = R_raw.T @ np.array([0, 0, 9.81])
                    raw_lin_acc = accel_meas_ms2 - gravity_sensor

                    euler_diff = ekf_euler - raw_euler
                    lin_acc_diff = ekf_lin_acc - raw_lin_acc

                    sum_sq_euler_diff += euler_diff ** 2
                    sum_sq_lin_acc_diff += lin_acc_diff ** 2
                    count += 1

                    if count % 1000 == 0:
                        rms_euler_diff = np.sqrt(sum_sq_euler_diff / count)
                        rms_lin_acc_diff = np.sqrt(sum_sq_lin_acc_diff / count)
                        rms_euler_diff_deg = np.degrees(rms_euler_diff)
                        print(f"\nSamples Processed: {count}")
                        print(f"RMS Euler Diff (deg): Roll: {rms_euler_diff_deg[0]:.2f}, "
                              f"Pitch: {rms_euler_diff_deg[1]:.2f}, Yaw: {rms_euler_diff_deg[2]:.2f}")
                        print(f"RMS Linear Acc Diff (m/s²): X: {rms_lin_acc_diff[0]:.2f}, "
                              f"Y: {rms_lin_acc_diff[1]:.2f}, Z: {rms_lin_acc_diff[2]:.2f}")

                    if count % 1000 == 0:
                        print("\n--- DEBUG DATA ---")
                        print(f"Raw Sensor Data: Accel: [{x_accel:.4f}, {y_accel:.4f}, {z_accel:.4f}], "
                              f"Gyro (°/s): [{x_gyro:.4f}, {y_gyro:.4f}, {z_gyro:.4f}]")
                        print(f"Bias Corrected Data: Accel (g): {accel_meas}, Gyro (rad/s): {gyro_meas_rad}")
                        print(f"dt: {dt:.4f}")
                        print(f"EKF Euler (deg): {np.degrees(ekf_euler)}")
                        print(f"Raw Euler (deg): {np.degrees(raw_euler)}")
                        print(f"EKF Lin Acc (m/s²): {ekf_lin_acc}")
                        print(f"Raw Lin Acc (m/s²): {raw_lin_acc}")
                        print("--- END DEBUG DATA ---")

    except KeyboardInterrupt:
        print("\nTerminating live data collection...")
    finally:
        ser.close()


if __name__ == "__main__":
    live_ekf_test()
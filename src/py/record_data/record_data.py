import time
import keyboard
import numpy as np
import serial
from scipy.spatial.transform import Rotation

from src.py.anal.compare_rotation import compare_rotations, rotation_difference_angle
from src.py.configs.config import SERIAL_PORT, BAUD_RATE, LABELS, CALIBRATION_ITERATIONS
from src.py.configs.run_config import config
from src.py.record_data.file_manager import FileManager
from src.py.sensor_calibration.calibrate import calibrate_sensors
from src.py.sensor_calibration.calibration_data import get_calibration_data
from src.py.sensor_calibration.ekf import EKF

record = config["record"]


def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Connected to ESP32. Press SPACE to start/stop recording.")

    # Calibration
    accel, gyro, accel_bias, gyro_bias = get_calibration_data(ser, CALIBRATION_ITERATIONS)
    calibrated_accel = [np.array(a) for a in accel]
    R_calib, gravity_mag = calibrate_sensors(calibrated_accel)
    print("Calibration complete!")
    print(f"Gravity magnitude: {gravity_mag}")

    # Initialize EKF
    ekf = EKF(dt=0.01, process_noise=0.01, accel_noise=0.5, gyro_noise=0.1)
    ekf.set_gravity_magnitude(gravity_mag)  # Set the correct gravity magnitude from calibration

    file_manager = None
    if record:
        file_manager = FileManager(ekf=True, linear_accel=True, eskf=True)  # Enable ESKF logging

    recording = False
    try:
        last_time = time.time()
        print_time = last_time
        while True:
            if record:
                if keyboard.is_pressed("space"):
                    if recording:
                        file_manager.finish_recording()
                        recording = False
                        print("Recording stopped")
                    else:
                        file_manager.setup_files()
                        recording = True
                        print("Recording started")
                    keyboard.wait("space")

                for key in LABELS:
                    if keyboard.is_pressed(key):
                        active_label = LABELS[key]
                        print(f"Active label: {active_label}")
                        file_manager.label = active_label  # Set label for file naming
                        while keyboard.is_pressed(key):
                            pass

            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                values = line.split('\t')
                if len(values) >= 6:
                    try:
                        x_accel, y_accel, z_accel, x_gyro, y_gyro, z_gyro = map(float, values[:6])
                    except Exception as e:
                        print(f"Error converting sensor data: {e}")
                        continue

                    # Apply calibration rotation to accelerometer data
                    raw_accel = np.array([x_accel, y_accel, z_accel])
                    raw_gyro = np.array([x_gyro, y_gyro, z_gyro])

                    # Apply calibration rotation to raw accelerometer data
                    rotated_accel = R_calib.apply(raw_accel)

                    # Correct gyro bias
                    corrected_gyro = raw_gyro - gyro_bias

                    # Calculate time step
                    current_time = time.time()
                    dt = current_time - last_time
                    last_time = current_time

                    # Skip if dt is too small to avoid numerical issues
                    if dt < 1e-5:
                        continue

                    # Update EKF
                    ekf_state = ekf.update(rotated_accel, corrected_gyro, dt)

                    # Get the current orientation from EKF
                    ekf_rotation = ekf.get_rotation()
                    ekf_quaternion = ekf.get_quaternion()  # [w, x, y, z]
                    ekf_euler = ekf.get_euler_angles()  # [roll, pitch, yaw]

                    # Get linear acceleration by removing gravity
                    R_ekf = ekf_rotation.as_matrix()
                    gravity_vector = R_ekf.T @ np.array([0, 0, gravity_mag])
                    linear_accel = rotated_accel - gravity_vector

                    # Every second, print the current state
                    if current_time - print_time > 1.0:
                        print_time = current_time
                        roll, pitch, yaw = np.degrees(ekf_euler)
                        print(f"Orientation (deg): Roll={roll:.1f}, Pitch={pitch:.1f}, Yaw={yaw:.1f}")

                        # Print gyro bias
                        gyro_bias_estimate = ekf.get_gyro_bias()
                        print(f"Gyro bias: {gyro_bias_estimate}")

                    # Record data if enabled
                    if record and recording:
                        data_dict = {
                            "raw": [x_accel, y_accel, z_accel, x_gyro, y_gyro, z_gyro],
                            "ekf_quaternion": [ekf_quaternion[0], ekf_quaternion[1], ekf_quaternion[2],
                                               ekf_quaternion[3]],
                            "ekf_euler": [ekf_euler[0], ekf_euler[1], ekf_euler[2]],
                            "linear_accel": [linear_accel[0], linear_accel[1], linear_accel[2]],
                            "gyro_bias": [gyro_bias_estimate[0], gyro_bias_estimate[1], gyro_bias_estimate[2]]
                        }
                        file_manager.update(data_dict)

    except KeyboardInterrupt:
        print("\nExiting...")
        if record:
            file_manager.finish_recording()
        ser.close()


if __name__ == "__main__":
    main()
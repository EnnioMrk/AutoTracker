import time
import keyboard
import numpy as np
import serial
from scipy.spatial.transform import Rotation

from src.py.anal.velocity import compare_velocity
from src.py.configs.config import SERIAL_PORT, BAUD_RATE, LABELS, CALIBRATION_ITERATIONS, VELOCITY_INTERVAL, \
    ax_bias, ay_bias, az_bias, gx_bias, gy_bias, gz_bias
from src.py.record_data.file_manager import FileManager
from src.py.sensor_calibration.calibrate import calibrate_sensors
from src.py.sensor_calibration.calibration_data import get_calibration_data

# Import both EKF and ESKF implementations
from src.py.sensor_calibration.ekf import EKF, ESKF

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Connected to ESP32. Press SPACE to start/stop recording.")

    dt = 0.01  # Example time step
    ekf = EKF(dt=dt)
    eskf = ESKF(dt=dt)

    # Calibration
    measurements = get_calibration_data(ser, CALIBRATION_ITERATIONS)
    R_calib, gravity_mag = calibrate_sensors(measurements)
    accel_bias = np.array([ax_bias, ay_bias, az_bias])
    gyro_bias = np.array([gx_bias, gy_bias, gz_bias])
    print("Calibration complete!")

    # Set EKF and ESKF initial conditions and biases
    ekf.set_gravity_magnitude(gravity_mag)
    ekf.set_biases(accel_bias=accel_bias, gyro_bias=gyro_bias)
    q_calib = R_calib.as_quat()
    q_ekf = np.concatenate(([q_calib[3]], q_calib[:3]))
    ekf.set_initial_orientation(q_ekf)

    eskf.set_gravity_magnitude(gravity_mag)
    eskf.set_biases(accel_bias=accel_bias, gyro_bias=gyro_bias)
    eskf.set_initial_orientation(q_ekf) # Initialize ESKF with same orientation

    file_manager = FileManager(ekf=True, linear_accel=True, eskf=True) # Enable ESKF logging

    recording = False
    v_0 = 0
    start_time = time.time()
    time_interval = np.array([])

    try:
        last_time = time.time()
        while True:
            if keyboard.is_pressed("space"):
                if recording:
                    file_manager.finish_recording()
                    recording = False
                else:
                    file_manager.setup_files()
                    recording = True
                keyboard.wait("space")

            for key in LABELS:
                if keyboard.is_pressed(key):
                    active_label = LABELS[key]
                    print(f"Active label: {active_label}")
                    file_manager.label = active_label # Set label for file naming
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

                    # Apply bias corrections
                    x_accel_corrected_ekf = x_accel - ekf.accel_bias[0]
                    y_accel_corrected_ekf = y_accel - ekf.accel_bias[1]
                    z_accel_corrected_ekf = z_accel - ekf.accel_bias[2]
                    x_gyro_corrected_ekf = x_gyro - ekf.gyro_bias[0]
                    y_gyro_corrected_ekf = y_gyro - ekf.gyro_bias[1]
                    z_gyro_corrected_ekf = z_gyro - ekf.gyro_bias[2]

                    imu_vector_ekf = [x_accel_corrected_ekf, y_accel_corrected_ekf, z_accel_corrected_ekf, x_gyro_corrected_ekf, y_gyro_corrected_ekf, z_gyro_corrected_ekf]
                    imu_vector_eskf = [x_accel, y_accel, z_accel, x_gyro, y_gyro, z_gyro]


                    current_time = time.time()
                    dt = current_time - last_time
                    last_time = current_time

                    # EKF Prediction and Update
                    ekf.predict(u=imu_vector_ekf, dt=dt)
                    ekf.update(np.array(imu_vector_ekf))
                    ekf_linear_accel = ekf.get_linear_acceleration()
                    ekf_pos = ekf.get_position()
                    ekf_vel = ekf.get_velocity()
                    ekf_quat = ekf.get_quaternion()

                    # ESKF Prediction and Update
                    eskf.predict(u=imu_vector_eskf, dt=dt)
                    eskf.update(np.array(imu_vector_eskf))
                    eskf_pos = eskf.get_position()
                    eskf_vel = eskf.get_velocity()
                    eskf_quat = eskf.get_quaternion()
                    eskf_gyro_bias = eskf.get_gyro_bias()
                    eskf_accel_bias = eskf.get_accel_bias()

                    ekf_euler = Rotation.from_quat( quat=ekf_quat, cls_1=None).as_euler("xyz", degrees=True)
                    eskf_euler = Rotation.from_quat( quat=eskf_quat, cls_1=None).as_euler("xyz", degrees=True)

                    ekf_velocity_magnitude = np.sqrt(ekf_vel[0]**2 + ekf_vel[1]**2 + ekf_vel[2]**2)
                    eskf_velocity_magnitude = np.sqrt(eskf_vel[0]**2 + eskf_vel[1]**2 + eskf_vel[2]**2)

                    data_dict = {
                        "raw": [x_accel, y_accel, z_accel, x_gyro, y_gyro, z_gyro],
                        "ekf": [ekf_pos[0], ekf_pos[1], ekf_pos[2], ekf_vel[0], ekf_vel[1], ekf_vel[2],
                                ekf_quat[0], ekf_quat[1], ekf_quat[2], ekf_quat[3]],
                        "eskf": [eskf_pos[0], eskf_pos[1], eskf_pos[2], eskf_vel[0], eskf_vel[1], eskf_vel[2],
                                 eskf_quat[0], eskf_quat[1], eskf_quat[2], eskf_quat[3],
                                 eskf_gyro_bias[0], eskf_gyro_bias[1], eskf_gyro_bias[2],
                                 eskf_accel_bias[0], eskf_accel_bias[1], eskf_accel_bias[2]] # Log ESKF biases too
                    }
                    file_manager.update(data_dict)

                    time_interval = np.append(time_interval, np.linalg.norm(ekf_vel)) # Using EKF vel for velocity comparison - can change to ESKF
                    check_time = time.time()
                    if check_time - start_time > VELOCITY_INTERVAL:
                        v_0 = compare_velocity(time_interval, v_0)
                        start_time = check_time
                        time_interval = np.array([])

                    print(f"EKF Pos: {ekf_pos}, Vel: {ekf_vel}, Ori: {ekf_quat[:3]}")
                    print(f"ESKF Pos: {eskf_pos}, Vel: {eskf_vel}, Ori: {eskf_quat[:3]}, Gyro Bias: {eskf_gyro_bias}, Accel Bias: {eskf_accel_bias}")


    except KeyboardInterrupt:
        print("\nExiting...")
        file_manager.finish_recording()
        ser.close()

if __name__ == "__main__":
    main()
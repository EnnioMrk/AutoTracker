import os
import keyboard
import numpy as np
import serial

from config import SERIAL_PORT, BAUD_RATE, LABELS
from sensor_calibration.calibration import VectorMeasurement, calibrate_sensors


def get_new_filename(active_label, raw=False):
    print("Creating file with label:", active_label)
    base_name = "sensor_data"
    extension = ".txt"

    index = 0
    if not active_label:
        while os.path.exists(
                f"{base_name}{'_raw' if raw else ''}{'_' + str(index) if index > 0 else ''}{extension}"
        ):
            index += 1
    else:
        while os.path.exists(f"{base_name}_{active_label}{'_raw' if raw else ''}{'_' + str(index) if index > 0 else ''}{extension}"):
            index += 1

    if active_label:
        return f"{base_name}_{active_label}{'_raw' if raw else ''}{'_' + str(index) if index > 0 else ''}{extension}"
    else:
        return f"{base_name}{'_raw' if raw else ''}{'_' + str(index) if index > 0 else ''}{extension}"

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Connected to ESP32. Press SPACE to start/stop recording.")

    recording = False
    file = None
    active_label = ""
    calibrate_index = 0
    measurements = []

    while (calibrate_index <= 1000):
        if ser.in_waiting > 0:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            parts = line.split("\t")
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            measurements.append(VectorMeasurement(x, y, z))
            calibrate_index += 1
        print(calibrate_index)

    r = calibrate_sensors(measurements, calibrate_index)

    try:
        while True:
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN:
                if event.name == "space":
                    if recording:
                        print(f"Stopping recording... Data saved in {filename}")
                        # Rename file
                        new_filename = get_new_filename(active_label)
                        file.close()  # Close the file when stopping

                        os.rename(filename, new_filename)
                        active_label = ""  # Reset active label
                        file = None
                        recording = False
                    else:
                        filename = "sensor_data.txt"
                        file_name = "sensor_data_raw.txt"
                        file = open(filename, "w", encoding="utf-8")
                        print(f"Started recording to {filename}...")
                        recording = True

                elif event.name in LABELS:
                    active_label = LABELS[event.name]
                    print(f"Active label: {active_label}")

                # Prevent multiple detections of one key press
                while keyboard.is_pressed("space"):
                    pass

            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if recording and file:
                    parts = line.split("\t")
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])

                    vector = np.array([x, y, z])
                    vector_transformed = r @ vector

                    new_x = vector_transformed[0]
                    new_y = vector_transformed[1]
                    new_z = vector_transformed[2]

                    line = f"{parts[0]}\t{new_x}\t{new_y}\t{new_z}"
                    print(line)
                    file.write(line + "\n")
                    file.flush()  # Ensure data is written immediately

    except KeyboardInterrupt:
        print("\nExiting...")
        if file:
            file.close()
        ser.close()

if __name__ == "__main__":
    main()
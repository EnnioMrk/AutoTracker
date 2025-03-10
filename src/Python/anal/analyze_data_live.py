import os

import keyboard
import serial

from analyze_z_deepseek import process_z_values, predict_type
from config import SERIAL_PORT, BAUD_RATE

def analyze_live_data(analyze_type=1, record=False):
    if analyze_type == 1:
        analyze_live_data_1(record)
    elif analyze_type == 2:
        analyze_live_data_2()

def get_new_filename():
    print("Creating file with label: sensor_data")
    base_name = "sensor_recording"
    extension = ".txt"

    index = 0
    while os.path.exists(f"{base_name}{'_' + str(index) if index > 0 else ''}{extension}"):
        index += 1

    return f"{base_name}{'_' + str(index) if index > 0 else ''}{extension}"

def analyze_live_data_1(record=False):
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Connected to ESP32. Press CTRL+C to stop recording.")
    z_values = []
    frequency_sum = 0
    frequency_count = 0
    file = None

    if record:
        filename = get_new_filename()
        file = open(filename, "w", encoding="utf-8")

    try:
        while True:
            if len(z_values) >= frequency_sum/frequency_count:
                features = process_z_values(z_values, frequency_sum/frequency_count)
                if features:
                    model = predict_type(features)
                    print(f"Prediction: {model}")

                z_values = []
                frequency_sum = 0
                frequency_count = 0

            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8").strip()
                parts = line.strip().split("\t")
                z_values.append(float(parts[1]))
                frequency_sum += float(parts[0])
                frequency_count += 1

                if record and file:
                    file.write(line + "\n")
                    file.flush()

    except KeyboardInterrupt:
        if file:
            file.close()

def analyze_live_data_2(record=False):
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Connected to ESP32. Press CTRL+C to stop recording.")
    z_values = []
    frequency_sum = 0
    frequency_count = 0
    file = None
    analyzing = False

    if record:
        filename = get_new_filename()
        file = open(filename, "w", encoding="utf-8")

    try:
        while True:
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN:
                if event.name == "space":
                    if analyzing:
                        features = process_z_values(z_values, frequency_sum/frequency_count)
                        if features:
                            model = predict_type(features)
                            print(f"Prediction: {model}")

                        # Reset variables
                        z_values = []
                        frequency_sum = 0
                        frequency_count = 0
                        analyzing = False
                    else:
                        analyzing = True

            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if analyzing:
                    parts = line.strip().split("\t")
                    z_values.append(float(parts[1]))
                    frequency_sum += float(parts[0])
                    frequency_count += 1
                if record and file:
                    file.write(line + "\n")
                    file.flush()  # Ensure data is written immediately


    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    analyze_live_data()


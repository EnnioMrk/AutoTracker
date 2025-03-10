import os
import keyboard
import serial

from config import SERIAL_PORT, BAUD_RATE, LABELS

def get_new_filename(active_label):
    print("Creating file with label:", active_label)
    base_name = "sensor_data"
    extension = ".txt"

    index = 0
    if not active_label:
        while os.path.exists(f"{base_name}{'_' + str(index) if index > 0 else ''}{extension}"):
            index += 1
    else:
        while os.path.exists(f"{base_name}_{active_label}{'_' + str(index) if index > 0 else ''}{extension}"):
            index += 1

    if active_label:
        return f"{base_name}_{active_label}{'_' + str(index) if index > 0 else ''}{extension}"
    else:
        return f"{base_name}{'_' + str(index) if index > 0 else ''}{extension}"

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Connected to ESP32. Press SPACE to start/stop recording.")

    recording = False
    file = None
    active_label = ""

    try:
        while True:
            if keyboard.is_pressed("space"):
                if recording:
                    print(f"Stopping recording... Data saved in {filename}")
                    new_filename = get_new_filename(active_label)
                    file.close()
                    os.rename(filename, new_filename)
                    active_label = ""
                    file = None
                    recording = False
                else:
                    filename = "sensor_data.txt"
                    file = open(filename, "w", encoding="utf-8")
                    print(f"Started recording to {filename}...")
                    recording = True

                # Prevent multiple detections of one key press
                while keyboard.is_pressed("space"):
                    pass

            for key in LABELS:
                if keyboard.is_pressed(key):
                    active_label = LABELS[key]
                    print(f"Active label: {active_label}")
                    while keyboard.is_pressed(key):
                        pass  # Prevent multiple detections

            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if recording and file:
                    file.write(line + "\n")
                    file.flush()

    except KeyboardInterrupt:
        print("\nExiting...")
        if file:
            file.close()
        ser.close()

if __name__ == "__main__":
    main()
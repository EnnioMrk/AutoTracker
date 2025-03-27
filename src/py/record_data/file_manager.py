import csv
import os
import keyboard
from src.py.configs.config import LABELS

class FileManager:
    data_path = os.path.abspath("../data")
    paths = {
        "raw": os.path.join(data_path, "raw"),
        "ekf": os.path.join(data_path, "ekf"),
        "eskf": os.path.join(data_path, "eskf")
    }

    def __init__(self, ekf=True, linear_accel=True, eskf=True):
        self.recording = False
        self._flags = {
            "raw": True,
            "ekf": ekf,
            "eskf": eskf
        }
        self.label = None
        self._files = {}
        self._csv_writers = {}
        self._files_closed = False
        # Add state variable to track if space was pressed in the previous check
        self._space_was_pressed = False

    # ... ( _get_file_name, setup_files, _write_data, update are likely okay) ...
    def _get_file_name(self, label, path):
        base_name = "data"
        extension = ".csv"
        index = 0
        name_with_label = f"{base_name}{'_' + label if label else ''}"
        while os.path.exists(os.path.join(self.paths[path], f"{name_with_label}{'_' + str(index) if index > 0 else ''}{extension}")):
            index += 1
        return f"{name_with_label}{'_' + str(index) if index > 0 else ''}{extension}"

    def setup_files(self):
        """Creates and initializes CSV files for enabled flags."""
        os.makedirs(FileManager.data_path, exist_ok=True)
        self._files_closed = False # Reset closed flag when setting up new files
        for key, enabled in self._flags.items():
            if enabled:
                os.makedirs(FileManager.paths[key], exist_ok=True)
                # Use a temporary name initially
                file_name = f'data_recording_{key}_temp.csv'
                file_path = os.path.join(FileManager.paths[key], file_name)
                try:
                    # Close existing file if somehow open (safety)
                    if key in self._files and not self._files[key].closed:
                         self._files[key].close()
                    self._files[key] = open(file_path, "w", newline="")
                    self._csv_writers[key] = csv.writer(self._files[key])
                    print(f"Opened temp file for {key}: {file_path}")
                except Exception as e:
                    print(f"Error opening file for {key}: {e}")
                    # Handle error, maybe disable this path
                    self._flags[key] = False


    def _write_data(self, data, path):
        """Writes data to the corresponding CSV file."""
        if path in self._csv_writers and not self._files_closed:
            try:
                writer = self._csv_writers[path]
                file = self._files[path]
                if not file.closed:
                    writer.writerow(data)
                    file.flush() # Ensure immediate write to disk
                else:
                     print(f"Warning: Attempted to write to closed file for '{path}'.")
            except Exception as e:
                print(f"Error writing data for {path}: {e}")

    def update(self, data_dict):
        # Removed the print("Updating") as it can be very noisy
        """Writes multiple data entries at once."""
        if not self.recording or self._files_closed:
             return # Don't write if not recording or files are closed

        for path, data in data_dict.items():
            # Check flag and if data is not None/empty
            if self._flags.get(path, False) and data:
                self._write_data(data, path)

    def finish_recording(self):
        """Closes all open file handles and renames files."""
        if self._files_closed:
            return

        print("Finishing recording...")
        self.recording = False # Ensure recording state is off

        closed_correctly = True
        for key, file in self._files.items():
            try:
                if not file.closed:
                    file.close()
                    print(f"Closed file for {key}.")
            except Exception as e:
                print(f"Error closing file for {key}: {e}")
                closed_correctly = False

        if closed_correctly:
             print("All temp files closed.")
        else:
             print("Warning: Some files might not have closed correctly.")


        for key, enabled in self._flags.items():
            if enabled and key in self._files:
                temp_name = f'data_recording_{key}_temp.csv'
                temp_path = os.path.join(FileManager.paths[key], temp_name)
                final_name = self._get_file_name(self.label, key)
                final_path = os.path.join(FileManager.paths[key], final_name)

                if os.path.exists(temp_path):
                    try:
                        os.rename(temp_path, final_path)
                        print(f"Renamed temp file for {key} to {final_name}")
                    except Exception as e:
                        print(f"Error renaming file for {key}: {e}. Temp file: {temp_path}")

        self._files.clear()
        self._csv_writers.clear()
        self._files_closed = True
        print("Recording finished and files processed.")


    def detect_keypress(self):
        """Detects key presses and updates the active label or toggles recording state."""
        space_is_currently_pressed = keyboard.is_pressed("space")

        if space_is_currently_pressed and not self._space_was_pressed:
            if self.recording:
                self.finish_recording()
                print("Recording stopped")
            else:
                self.setup_files()
                self.recording = True
                print("Recording started")

        # Update the state for the next check
        self._space_was_pressed = space_is_currently_pressed

        for key in LABELS:
            if keyboard.is_pressed(key):
                active_label = LABELS[key]
                if self.label != active_label: # Optional: Only print if label changes
                    print(f"Active label: {active_label}")
                    self.label = active_label

    def __del__(self):
        """Ensures all files are closed when the object is destroyed."""
        self.finish_recording()
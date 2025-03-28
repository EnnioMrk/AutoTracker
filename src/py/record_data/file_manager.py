# --- FileManager.py ---

import csv
import os
from typing import Optional, Dict, Any

import keyboard
import json
import numpy as np
from src.py.configs.config import LABELS
from scipy.spatial.transform import Rotation

class NumpyEncoder(json.JSONEncoder):
    """ Special JSON encoder for numpy types and Rotation objects """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Rotation):
             # Always save Rotation objects as quaternions [x, y, z, w]
             return obj.as_quat().tolist()
        return json.JSONEncoder.default(self, obj)

class FileManager:
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    paths = {
        "raw": os.path.join(data_path, "raw")
        # Add other paths like "processed" if needed later
    }

    def __init__(self):
        self.recording = False
        self._flags = {"raw": True} # Only handle raw CSV saving here
        self.label = None
        self._files: Dict[str, Any] = {}
        self._csv_writers: Dict[str, Any] = {}
        self._files_closed = True # Start as closed
        self._space_was_pressed = False
        # Store the path for the *last* successfully renamed CSV file of this instance
        self.last_final_csv_path: Optional[str] = None
        # Path to the temporary file being written to
        self._current_temp_csv_path: Dict[str, Optional[str]] = {}

    def _get_file_name(self, label: Optional[str], path_key: str) -> str:
        """Generates a unique final CSV filename."""
        base_name = "data"
        extension = ".csv"
        index = 0
        target_dir = self.paths.get(path_key)
        if not target_dir:
            print(f"Error: Path key '{path_key}' not found.")
            return f"{base_name}_error_{path_key}.csv" # Return error filename

        # Ensure label is string or None
        label_str = str(label) if label is not None else ''
        name_with_label = f"{base_name}{'_' + label_str if label_str else ''}"

        while True:
            prospective_name = f"{name_with_label}{'_' + str(index) if index > 0 else ''}{extension}"
            if not os.path.exists(os.path.join(target_dir, prospective_name)):
                return prospective_name
            index += 1

    def setup_files(self):
        """Creates and opens temporary CSV files for recording."""
        # Ensure base data path exists
        os.makedirs(FileManager.data_path, exist_ok=True)
        self._files_closed = False # Mark as open/active
        self.last_final_csv_path = None # Reset last saved path for new session start
        self._current_temp_csv_path = {} # Reset temp paths

        print("Setting up temporary files for recording...")
        for key, enabled in self._flags.items():
            if enabled:
                target_dir = self.paths.get(key)
                if not target_dir: continue # Skip if path key invalid
                os.makedirs(target_dir, exist_ok=True)

                # Define temp file path
                temp_file_name = f'data_recording_{key}_temp_{os.getpid()}.csv' # Add PID for uniqueness if multiple instances run
                temp_file_path = os.path.join(target_dir, temp_file_name)
                self._current_temp_csv_path[key] = temp_file_path

                try:
                    # Ensure any lingering old file handle is closed
                    if key in self._files and self._files[key] and not self._files[key].closed:
                         self._files[key].close()
                    # Open new temp file
                    self._files[key] = open(temp_file_path, "w", newline="")
                    self._csv_writers[key] = csv.writer(self._files[key])
                    print(f"  Opened temp file: {temp_file_path}")
                except Exception as e:
                    print(f"  Error opening temp file for '{key}': {e}")
                    self._current_temp_csv_path[key] = None # Mark as failed
                    self._flags[key] = False # Disable this type for current run

    def _write_data(self, data: list, path_key: str):
        """Writes a data row to the specified temporary CSV file."""
        if path_key not in self._csv_writers or path_key not in self._files:
            # print(f"Debug: Writer/File not ready for {path_key}") # Debug line
            return # Skip if setup failed or key invalid

        if not self._files_closed:
            try:
                writer = self._csv_writers[path_key]
                file = self._files[path_key]
                if file and not file.closed:
                    writer.writerow(data)
                # else: print(f"Debug: File {path_key} closed unexpectedly during write.") # Debug
            except Exception as e:
                print(f"Error writing data for {path_key}: {e}")

    def update(self, data_dict: Dict[str, list]):
        """Writes data rows for enabled flags if recording is active."""
        if not self.recording or self._files_closed:
             return

        for path_key, data in data_dict.items():
            # Check if flag enabled, data exists, and temp file was set up
            if self._flags.get(path_key, False) and data and self._current_temp_csv_path.get(path_key):
                 if path_key == 'raw' and len(data) != 11:
                      print(f"Warning: 'raw' data update expected 11 elements, got {len(data)}. Skipping.")
                      continue
                 self._write_data(data, path_key)

    def finish_recording(self):
        """
        Closes the current temporary CSV files and renames them to their final names.
        Does NOT save JSON calibration data. Stores the path of the last saved CSV.
        """
        if self._files_closed:
            # print("Debug: finish_recording called but files already closed.") # Debug
            return # Already processed

        print("Finishing current recording segment (closing/renaming CSV)...")
        self.recording = False # Stop accepting new data via update

        # --- Close Temp CSV Files ---
        for key, file in self._files.items():
            try:
                if file and not file.closed:
                    file.close()
                    print(f"  Closed temp file for '{key}'.")
            except Exception as e:
                print(f"  Error closing temp file for '{key}': {e}")

        # --- Rename Temp CSV Files to Final Names ---
        # Reset last saved path before potentially setting it
        self.last_final_csv_path = None
        for key, enabled in self._flags.items():
            if enabled and self._current_temp_csv_path.get(key):
                temp_csv_path = self._current_temp_csv_path[key]
                target_dir = self.paths.get(key)

                if not target_dir or not os.path.exists(temp_csv_path):
                     print(f"  Warning: Temp CSV for '{key}' not found or invalid path. Cannot rename.")
                     continue

                # Determine final unique name in the target directory
                final_csv_name = self._get_file_name(self.label, key)
                final_csv_path = os.path.join(target_dir, final_csv_name)

                try:
                    os.rename(temp_csv_path, final_csv_path)
                    print(f"  Renamed temp CSV for '{key}' to '{final_csv_name}'")
                    # Store the path of the successfully renamed file
                    self.last_final_csv_path = final_csv_path
                except Exception as e:
                    print(f"  Error renaming CSV for '{key}': {e}. Temp file left: {temp_csv_path}")
                    # If renaming fails, last_final_csv_path remains None or retains previous value

        # --- Clean up internal state for the closed segment ---
        self._files.clear()
        self._csv_writers.clear()
        self._current_temp_csv_path.clear()
        self._files_closed = True # Mark files for this segment as done
        print("CSV processing for current segment complete.")


    def save_calibration_json(self, calibration_data: Optional[Dict]):
        """
        Saves the provided calibration data as a JSON file next to the
        *last successfully saved CSV file*. Should be called once at the very end.
        """
        if self.last_final_csv_path is None:
             print("Info: No CSV file was successfully saved in the last segment, skipping JSON save.")
             return
        if calibration_data is None:
             print("Info: No calibration data provided, skipping JSON save.")
             return

        # Construct JSON path based on the last CSV path
        json_file_path = os.path.splitext(self.last_final_csv_path)[0] + '.json'

        print(f"Attempting to save final calibration data to: {json_file_path}")
        try:
            # Prepare data (handle Rotation object specifically)
            calibration_data_to_save = {}
            for key, value in calibration_data.items():
                 if key == 'R_gravity' and isinstance(value, Rotation):
                      # Store the Rotation object as its quaternion representation
                      calibration_data_to_save['R_gravity_quat'] = value.as_quat().tolist()
                 else:
                      calibration_data_to_save[key] = value

            # Save using custom encoder for numpy types
            with open(json_file_path, 'w') as f:
                json.dump(calibration_data_to_save, f, indent=4, cls=NumpyEncoder)
            print(f"Successfully saved calibration JSON: {json_file_path}")

        except TypeError as e:
             print(f"Error serializing calibration data to JSON: {e}")
             # Optionally print the problematic data structure for debugging
             # print("Data causing error:", calibration_data_to_save)
        except Exception as e:
            print(f"Error writing JSON calibration file '{json_file_path}': {e}")


    def detect_keypress(self):
        """Detects key presses for recording toggle and labeling."""
        try:
            space_is_currently_pressed = keyboard.is_pressed("space")

            # Toggle recording state on Spacebar press (rising edge)
            if space_is_currently_pressed and not self._space_was_pressed:
                if self.recording:
                    # Currently recording -> Stop this segment
                    self.finish_recording() # Close/rename CSV for this segment
                    print("Recording segment stopped via Spacebar.")
                    # self.recording is set to False inside finish_recording
                else:
                    # Currently stopped -> Start new segment
                    # Reset label for new segment?
                    self.label = None
                    print("Label reset for new segment.")
                    self.setup_files() # Open new temp files
                    # Check if setup was successful for at least one file type
                    if any(self._current_temp_csv_path.get(key) for key in self._flags if self._flags[key]):
                         self.recording = True # Mark as recording
                         print("Recording segment started via Spacebar. Press label keys...")
                    else:
                         print("Error: Failed to set up any temporary files. Cannot start recording.")
                         self._files_closed = True # Ensure state reflects failure

            self._space_was_pressed = space_is_currently_pressed # Update state

            # Check for label keys ONLY if currently recording
            if self.recording:
                for key_char, label_name in LABELS.items():
                    if keyboard.is_pressed(key_char):
                        if self.label != label_name: # Update label if changed
                            print(f"Active label set to: {label_name}")
                            self.label = label_name
                        break # Only process one label key per check cycle

        except ImportError:
             # Handle keyboard library not available
             if not hasattr(self, '_keyboard_import_error_shown'):
                  print("\nWarning: 'keyboard' library not found/usable. Live labeling and Spacebar control disabled.\n")
                  self._keyboard_import_error_shown = True
             pass # Allow program to continue without keyboard input
        except Exception as e:
             # Handle other keyboard errors
            if not hasattr(self, '_keyboard_generic_error_shown'):
                print(f"\nWarning: Error checking keyboard input: {e}. Live controls may fail.\n")
                self._keyboard_generic_error_shown = True
            pass


    def __del__(self):
        """Ensures files are attempted to be closed on object deletion."""
        # Call finish_recording to close/rename any lingering temp file
        # JSON saving is NOT done here, only in DataCollector.stop
        if not self._files_closed:
             print("FileManager destructor: Calling finish_recording for cleanup...")
             self.finish_recording()
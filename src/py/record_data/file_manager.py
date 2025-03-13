import csv
import os


class FileManager:
    data_path = os.path.abspath("../data")
    paths = {
        "raw": os.path.join(data_path, "raw"),
        "filtered_accel": os.path.join(data_path, "filtered_accel"),
        "ekf": os.path.join(data_path, "ekf"),
        "eskf": os.path.join(data_path, "eskf")
    }

    def __init__(self, ekf=True, linear_accel=True, eskf=True):
        self._flags = {
            "raw": True,
            "filtered_accel": linear_accel,
            "ekf": ekf,
            "eskf": eskf # Added eskf flag
        }
        self.label = None
        self._files = {}
        self._csv_writers = {}
        self._files_closed = False

    # ... (rest of FileManager class - _get_file_name, setup_files, _write_data, update, finish_recording, __del__ - same as before) ...
    def _get_file_name(self, label, path):
        base_name = "data"
        extension = ".csv"
        index = 0

        # Construct base filename
        name_with_label = f"{base_name}{'_' + label if label else ''}"

        # Ensure unique filename
        while os.path.exists(os.path.join(self.paths[path], f"{name_with_label}{'_' + str(index) if index > 0 else ''}{extension}")):
            index += 1

        return f"{name_with_label}{'_' + str(index) if index > 0 else ''}{extension}"

    def setup_files(self):
        """Creates and initializes CSV files for enabled flags."""
        os.makedirs(FileManager.data_path, exist_ok=True)  # Ensure base directory exists

        for key, enabled in self._flags.items():
            if enabled:
                os.makedirs(FileManager.paths[key], exist_ok=True)
                file_name = f'data_recording_{key}'  # Temporary file name
                file_path = os.path.join(FileManager.paths[key], file_name)

                self._files[key] = open(file_path, "w", newline="")
                self._csv_writers[key] = csv.writer(self._files[key])

    def _write_data(self, data, path):
        """Writes data to the corresponding CSV file."""
        if path in self._csv_writers:
            self._csv_writers[path].writerow(data)
            self._files[path].flush()  # Ensure immediate write to disk
        else:
            print(f"Warning: Attempted to write to '{path}', but it's not enabled.")

    def update(self, data_dict):
        """Writes multiple data entries at once."""
        for path, data in data_dict.items():
            if self._flags.get(path, False) and data:  # Ensure the path is enabled
                self._write_data(data, path)

    def finish_recording(self):
        """Closes all open file handles."""
        if self._files_closed:
            return  # Prevents double execution

        for file in self._files.values():
            file.close()

        for key, enabled in self._flags.items():
            if enabled and key in self._files:  # Check if the file was opened
                temp_path = os.path.join(FileManager.paths[key], f'data_recording_{key}')
                final_path = os.path.join(FileManager.paths[key], self._get_file_name(self.label, key))

                if os.path.exists(temp_path):
                    os.rename(temp_path, final_path)

        self._files.clear()
        self._csv_writers.clear()
        self._files_closed = True  # Mark files as closed

    def __del__(self):
        """Ensures all files are closed when the object is destroyed."""
        self.finish_recording()
# --- DataCollector.py ---

import time
import traceback

from src.py.configs.config import SERIAL_PORT, BAUD_RATE
from src.py.configs.filter_config import FilterConfig
from src.py.configs.data_collector import MainConfig
from src.py.data_analysis.visualizer import DataVisualizer
from src.py.record_data.serial_reader import SerialReader, SerialConnectionError
from src.py.data_analysis.data_processor import DataProcessor
from src.py.record_data.file_manager import FileManager # Import modified FileManager
from src.py.sensor_calibration.calibrator import SensorCalibrator


class DataCollector:
    """
    Orchestrates sensor connection, calibration, data collection, processing,
    and manages recording segments and final calibration saving.
    """
    def __init__(self, config: MainConfig = None, filter_config: FilterConfig = None, visualizer: DataVisualizer = None):
        """ Initializes the DataCollector. """
        self.config = config or MainConfig()
        self.filter_config = filter_config or FilterConfig()
        self.visualizer = visualizer

        # Instantiate components
        self.file_manager = FileManager() # Instantiate modified FileManager
        try:
             self.ser = SerialReader(SERIAL_PORT, BAUD_RATE)
        except ImportError:
             print("Error: Could not import 'serial'. Please install pyserial.")
             self.ser = None
        except Exception as e:
             print(f"Error initializing SerialReader: {e}")
             self.ser = None

        self.data_processor = DataProcessor(self.file_manager, self.filter_config, self.config, self.visualizer)
        self.calibrator = SensorCalibrator(self.config, self.ser, self.data_processor)

        self.is_running = False # Overall collector is active
        self.run = True # Loop control flag

    # --- calibrate method remains the same ---
    def calibrate(self) -> bool:
        """ Performs the full sensor calibration sequence. """
        if self.ser is None:
             print("Error: Serial reader not initialized. Cannot calibrate.")
             return False
        print("Starting calibration process...")
        try:
            # 1. Collect stationary data
            print("Collecting stationary measurements for bias and gravity...")
            accel_measurements, gyro_measurements = self.calibrator.collect_stationary_measurements()

            # 2. Perform bias calibration
            print("Calibrating stationary bias...")
            if not self.calibrator.calibrate_stationary_bias():
                 print("Stationary bias calibration failed.")
                 return False

            # 3. Perform gravity alignment
            print("Calibrating gravity alignment...")
            if not self.calibrator.calibrate_gravity_alignment(accel_measurements):
                 print("Gravity alignment calibration failed.")
                 return False

            # 4. Set parameters in DataProcessor
            print("Setting calibration parameters in DataProcessor...")
            self.data_processor.set_calibration(
                 accel_bias=self.calibrator.accel_bias,
                 gyro_bias=self.calibrator.gyro_bias,
                 R_gravity=self.calibrator.gravity_rotation,
                 gravity_mag=self.calibrator.gravity_magnitude
            )

            print("-" * 20)
            print("Calibration Complete!")
            # (Optional: Print calibrated values)
            print("-" * 20)
            return True

        except RuntimeError as e: print(f"Calibration failed during data collection: {e}"); return False
        except SerialConnectionError as e: print(f"Serial Error during calibration: {e}"); return False
        except Exception as e: print(f"An unexpected error occurred during calibration: {e}"); traceback.print_exc(); return False


    # --- start method remains the same ---
    def start(self) -> bool:
        """ Connects to serial and calibrates. Sets running state. """
        if self.is_running: print("Data collection is already running."); return True
        if self.ser is None: print("Error: Serial Reader not available."); return False

        print("Attempting to connect to serial port...")
        try:
             self.ser.connect()
             print(f"Successfully connected to {SERIAL_PORT}.")
        except SerialConnectionError as e: print(f"Failed to connect to serial port: {e}"); return False
        except Exception as e: print(f"An unexpected error during serial connection: {e}"); return False

        if not self.calibrate():
            print("Calibration failed. Aborting start."); self.ser.close(); return False

        self.is_running = True; self.run = True
        print("\n-------------------------------------------")
        print("Data collection started successfully.")
        if self.config.record: print("Recording enabled. Spacebar toggles segments. Label keys: a, s, l, d.")
        print("Press Ctrl+C to stop collection.")
        print("-------------------------------------------\n")
        return True

    # --- collect_data method remains the same ---
    def collect_data(self):
        """ Reads and processes a single data point, handles recording logic. """
        if not self.is_running or self.ser is None: time.sleep(0.1); return None

        if self.config.record: self.file_manager.detect_keypress() # Detect start/stop/label

        try: raw_data = self.ser.read_data()
        except SerialConnectionError as e: print(f"Serial Error during data read: {e}. Stopping."); self.run = False; return None
        except Exception as e: print(f"Unexpected error reading from serial: {e}"); time.sleep(0.1); return None # Skip iteration

        if raw_data is None: return None # No new data

        if len(raw_data) == 7:
            ax, ay, az, gx, gy, gz, dt = raw_data
            acc = [ax, ay, az]; gyro = [gx, gy, gz]
            processed_result = self.data_processor.process(gyro, acc, dt)
            if processed_result is True: print("Stopping based on processor signal."); self.run = False # Check for visualizer stop signal
            return raw_data # Return raw data
        else:
            print(f"Warning: Expected 7 values, received {len(raw_data)}. Skipping."); return None


    # --- Modified stop method ---
    def stop(self):
        """
        Stops the data collection process gracefully:
        1. Finalizes the last CSV recording segment.
        2. Saves the calibration data as JSON next to the last CSV.
        3. Closes the serial port.
        4. Handles visualization.
        """
        # Prevent multiple stop executions
        if not self.is_running and not self.ser:
             return # Already stopped or never fully started

        print("\nStopping data collection...")
        # Immediately mark as not running to prevent race conditions
        self.is_running = False
        self.run = False # Signal any external loops

        # --- 1. Finalize the LAST CSV Recording Segment ---
        # This closes and renames the currently open temp file (if any)
        # and updates file_manager.last_final_csv_path
        if self.config.record:
             # No need to pass calibration data here anymore
             self.file_manager.finish_recording()

        # --- 2. Save Calibration JSON ---
        # This happens *after* the last CSV is potentially renamed
        if self.config.record:
            calibration_info = None
            if self.data_processor.calibrated:
                # Retrieve final calibration state
                calibration_info = {
                    "accel_bias": self.data_processor.accel_bias,
                    "gyro_bias": self.data_processor.gyro_bias,
                    "R_gravity": self.data_processor.R_gravity, # Pass Rotation object
                    "gravity_mag": self.data_processor.gravity_mag,
                }
                print("Attempting to save final calibration data...")
            else:
                 print("Warning: Calibration not marked complete. Cannot save calibration JSON.")

            # Call the dedicated JSON saving method in FileManager
            self.file_manager.save_calibration_json(calibration_info)

        # --- 3. Close Serial Connection ---
        if self.ser:
             self.ser.close()

        print("Data collection stopped successfully.")

        # --- 4. Handle Visualization ---
        if self.config.visualize and self.visualizer:
            print("Creating graphs (if data was collected)...")
            try:
                 if hasattr(self.visualizer, 'timestamps') and self.visualizer.timestamps:
                      self.visualizer.create_graphs()
                 else: print("No data collected for visualization.")
            except Exception as e: print(f"Error during graph creation: {e}")

# --- main function remains the same ---
def main():
    """ Main function to run the data collection process. """
    print("Initializing Data Collector...")
    main_config = MainConfig(record=True , visualize=False)
    filter_conf = FilterConfig()
    visualizer = DataVisualizer(max_iterations=20000)

    collector = DataCollector(config=main_config, filter_config=filter_conf, visualizer=visualizer)

    try:
        if collector.start():
            while collector.run:
                collector.collect_data()
        else: print("Failed to start data collector.")
    except KeyboardInterrupt: print("\nCtrl+C detected. Stopping...")
    except Exception as e: print(f"\nUnexpected error in main loop: {e}"); traceback.print_exc()
    finally:
        collector.stop()
        print("Program finished.")

if __name__ == "__main__":
    main()
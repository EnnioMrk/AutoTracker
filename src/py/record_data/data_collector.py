from src.py.configs.config import SERIAL_PORT, BAUD_RATE
from src.py.configs.filter_config import FilterConfig
from src.py.configs.data_collector import MainConfig
from src.py.record_data.serial_reader import SerialReader
from src.py.data_analysis.data_processor import DataProcessor
from src.py.record_data.file_manager import FileManager
from src.py.sensor_calibration.calibrator import SensorCalibrator


class DataCollector:
    def __init__(self, config=None, filter_config=None):
        """
        Initialize the sensor data collector.
        """
        self.config = config or MainConfig()
        self.filter_config = filter_config or FilterConfig()

        self.file_manager = FileManager()
        self.ser = SerialReader(SERIAL_PORT, BAUD_RATE)
        self.data_processor = DataProcessor(self.file_manager, self.filter_config)
        self.calibrator = SensorCalibrator(self.config, self.ser, self.data_processor)

        self.is_running = False

    def calibrate(self):
        """
        Perform sensor calibration.

        :return: True if calibration is successful, False otherwise
        """
        accel_measurements, _ = self.calibrator.collect_stationary_measurements()
        success = self.calibrator.perform_full_calibration(accel_measurements)

        if success:
            print("Calibration complete!")
            print(f"Gravity magnitude: {self.data_processor.gravity_mag}")
            print(f"Accel Bias: {self.data_processor.accel_bias}, Gyro Bias: {self.data_processor.gyro_bias}")
        else:
            print("Calibration failed.")

        return success

    def start(self):
        """
        Start sensor data collection.
        """
        if self.is_running:
            print("Data collection is already running.")
            return False

        # Connect to serial port
        self.ser.connect()

        # Perform calibration
        if not self.calibrate():
            self.ser.close()
            return False

        self.is_running = True
        print("Data collection started.")
        return True

    def collect_data(self):
        """
        Collect a single data point.

        :return: Processed data or None if collection fails
        """
        if not self.is_running:
            print("Start data collection first.")
            return None

        # Detect keypress for recording if enabled
        if self.config.record:
            self.file_manager.detect_keypress()

        # Read data from serial port
        data = self.ser.read_data()
        if data is None:
            return None

        ax, ay, az, gx, gy, gz, dt = data
        acc = [ax, ay, az]
        gyro = [gx, gy, gz]

        # Process and potentially record data
        self.data_processor.process(gyro, acc, dt)

        return data

    def stop(self):
        """
        Stop sensor data collection.
        """
        if not self.is_running:
            return

        # Stop recording if enabled
        if self.config.record:
            self.file_manager.finish_recording()

        # Close serial connection
        self.ser.close()

        self.is_running = False
        print("Data collection stopped.")


def main():
    # Example usage
    collector = DataCollector()
    try:
        # Start collection
        if collector.start():
            # Collect data points manually
            while True:
                collector.collect_data()
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop()


if __name__ == "__main__":
    main()
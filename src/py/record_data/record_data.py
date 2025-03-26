from src.py.configs.config import SERIAL_PORT, BAUD_RATE
from src.py.configs.run_config import MainConfig
from src.py.record_data.serial_reader import SerialReader
from src.py.data_analysis.data_processor import DataProcessor
from src.py.record_data.file_manager import FileManager
from src.py.sensor_calibration.calibrator import SensorCalibrator

config = MainConfig()
ser = SerialReader(SERIAL_PORT, BAUD_RATE)
data_processor = DataProcessor(config)
calibrator = SensorCalibrator(config, ser, data_processor)
record = config.record


def main():
    ser.connect()

    # Stationary Calibration
    if not calibrator.setup_stationary():
        # Temporary fix for calibration failure, because we only have stationary calibration
        print("Calibration failed.")
        ser.close()
        return

    print("Calibration complete!")
    print(f"Gravity magnitude: {data_processor.gravity_mag}")
    print(f"Accel Bias: {data_processor.accel_bias}, Gyro Bias: {data_processor.gyro_bias}")

    file_manager = None
    if record:
        file_manager = FileManager()

    recording = False
    try:
        while True:
            if record:
                file_manager.detect_keypress()

            # Read data from serial port
            data = ser.read_data()
            if data is None:
                continue

            ax, ay, az, gx, gy, gz, dt = data

            acc = [ax, ay, az]
            gyro = [gx, gy, gz]

            # Record data if enabled
            data_processor.process(gyro, acc, dt)

    except KeyboardInterrupt:
        print("\nExiting...")
        if record:
            file_manager.finish_recording()
        ser.close()

if __name__ == "__main__":
    main()
import serial

class SerialReader:
    """Handles serial communication for reading sensor data."""

    def __init__(self, port, baud_rate, timeout=1):
        self.port_name = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_port = None  # Initialize serial_port to None
        self._closed = False

    def connect(self):
        """Connects to the serial port."""
        try:
            self.serial_port = serial.Serial(self.port_name, self.baud_rate)
            print(f"Connected to serial port: {self.port_name}")
        except serial.SerialException as e:
            raise SerialConnectionError(f"Could not connect to serial port {self.port_name}: {e}")

    def read_data(self):
        """Reads and processes data from the serial port."""
        if self.serial_port is None:
            raise SerialConnectionError("Serial port is not connected. Call connect() first.")
        if self.serial_port.in_waiting > 0:
            line = self.serial_port.readline().decode("utf-8", errors="ignore").strip()
            values = line.split('\t')
            if len(values) >= 7:
                try:
                    x_accel, y_accel, z_accel, x_gyro, y_gyro, z_gyro, dt = map(float, values[:7])
                    return x_accel, y_accel, z_accel, x_gyro, y_gyro, z_gyro, dt
                except Exception as e:
                    print(f"Error converting sensor data: {e}")
                    return None
        return None

    def close(self):
        """Closes the serial port connection."""
        if self._closed:
            return
        if self.serial_port:
            self.serial_port.close()
            print(f"Serial port {self.port_name} closed.")
            self.serial_port = None  # Reset to None after closing
            self._closed = True

    def __del__(self):
        """Ensures the serial port is closed when the object is deleted."""
        self.close()

class SerialConnectionError(Exception):
    """Custom exception for serial connection errors."""
    pass
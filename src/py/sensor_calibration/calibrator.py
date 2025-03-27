from typing import Any, Optional, Tuple
import numpy as np
from numpy import ndarray, dtype, float64, floating
from scipy.spatial.transform import Rotation


class SensorCalibrator:
    """
    A comprehensive sensor calibration class for processing and aligning sensor data.

    Handles multiple calibration stages:
    1. Stationary bias calibration
    2. Gravity alignment
    3. Driving direction alignment
    """

    def __init__(self,
                 config,
                 serial_reader,
                 data_processor,
                 stationary_iterations: int = 100,
                 driving_samples: int = 500,
                 movement_detection_threshold: int = 30):
        """
        Initialize the SensorCalibrator.

        Args:
            config: Configuration object
            serial_reader: Serial data reader
            data_processor: Data processing utility
            stationary_iterations: Number of samples for stationary calibration
            driving_samples: Number of samples to collect during driving
            movement_detection_threshold: Consecutive non-stationary readings to detect movement
        """
        self.config = config
        self.serial_reader = serial_reader
        self.data_processor = data_processor

        # Calibration parameters
        self.stationary_iterations = stationary_iterations
        self.driving_samples = driving_samples
        self.movement_detection_threshold = movement_detection_threshold

        # Calibration results
        self.accel_bias = None
        self.gyro_bias = None
        self.gravity_rotation = None
        self.driving_rotation = None
        self.gravity_magnitude = None


    def collect_stationary_measurements(self) -> tuple[list[list[Any]], list[list[Any]]]:
        """
        Collect acceleration measurements during a stationary period.

        Returns:
            List of acceleration measurements
        """
        print("Collecting stationary measurements...")

        accel_measurements = []
        gyro_measurements = []
        not_stationary_count = 0
        iteration = 0

        while iteration < self.stationary_iterations:
            # Prevent infinite loop if sensor is constantly moving
            if not_stationary_count > 100:
                print("Sensor is not stationary. Please stop moving.")
                raise RuntimeError("Unable to collect stationary measurements")

            # Read sensor data
            data = self.serial_reader.read_data()
            if data is None:
                continue

            ax, ay, az, gx, gy, gz, dt = data

            if self.data_processor.calibrated:
                # Check if stationary
                stationary = self.data_processor.is_stationary(
                    [ax, ay, az], [gx, gy, gz]
                )

                if not stationary:
                    not_stationary_count += 1
                    continue

            # Collect measurements
            accel_measurements.append([ax, ay, az])
            gyro_measurements.append([gx, gy, gz])

            iteration += 1

        print(f"Collected {len(accel_measurements)} stationary measurements.")
        return accel_measurements, gyro_measurements

    @staticmethod
    def compute_trimmed_mean(measurements: list[list[Any]],
                             trim_percentage: float = 0.1) -> tuple[ndarray[tuple[int], dtype[Any]], float] | tuple[
        ndarray[tuple[int, ...], dtype[float64]] | ndarray[tuple[int, ...], dtype[Any]], floating[Any]]:
        """
        Compute the average vector using a trimmed mean approach.

        Args:
            measurements: List of measurement vectors
            trim_percentage: Percentage of extreme values to remove

        Returns:
            Tuple of (normalized vector, magnitude)
        """
        if not 0.0 <= trim_percentage <= 0.5:
            raise ValueError("trim_percentage must be between 0.0 and 0.5")

        # Convert measurements to a numpy array
        vectors = np.array(measurements)
        num_measurements = vectors.shape[0]

        if num_measurements == 0:
            return np.zeros(3, dtype=np.float64), 0.0

        # Trim and compute mean for each component
        trimmed_components = [
            np.sort(vectors[:, i])[
            int(num_measurements * trim_percentage):
            int(num_measurements * (1 - trim_percentage))
            ]
            for i in range(3)
        ]

        # Calculate mean vector
        mean_vec = np.array([np.mean(comp) for comp in trimmed_components])

        # Compute magnitude and normalize
        gravity_mag = np.linalg.norm(mean_vec)
        g_normalized = mean_vec / gravity_mag if gravity_mag > 0 else np.array([0, 0, 1])

        return g_normalized, gravity_mag

    @staticmethod
    def compute_rotation_matrix(source_vector: ndarray,
                                target_vector: Optional[ndarray] = None) -> ndarray:
        """
        Compute rotation matrix to align vectors.

        Args:
            source_vector: Vector to be rotated
            target_vector: Vector to align with (defaults to [0, 0, 1])

        Returns:
            3x3 rotation matrix
        """
        # If no target vector is provided, use world up [0, 0, 1]
        if target_vector is None:
            target_vector = np.array([0, 0, 1], dtype=np.float64)

        # Normalize input vectors
        source_normalized = source_vector / np.linalg.norm(source_vector)
        target_normalized = target_vector / np.linalg.norm(target_vector)

        # Compute cross-product to find rotation axis
        v = np.cross(source_normalized, target_normalized)
        v_norm = np.linalg.norm(v)

        # If vectors are nearly parallel, return identity matrix
        if v_norm < 1e-6:
            return np.identity(3)

        # Normalize rotation axis
        v = v / v_norm

        # Compute a rotation angle
        dot_val = np.dot(source_normalized, target_normalized)
        theta = np.arccos(np.clip(dot_val, -1.0, 1.0))

        # Compute skew-symmetric matrix
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])

        # Rodriguez rotation formula
        R = np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        return R

    def calibrate_stationary_bias(self) -> bool:
        """
        Calibrate sensor bias during stationary condition.

        Returns:
            Boolean indicating successful calibration
        """
        print("Starting stationary bias calibration...")

        accel_measurements, gyro_measurements = self.collect_stationary_measurements()

        # Compute biases
        self.accel_bias = np.mean(accel_measurements, axis=0)
        self.gyro_bias = np.mean(gyro_measurements, axis=0)

        # Update data processor
        self.data_processor.update_bias(self.gyro_bias, self.accel_bias)#
        self.data_processor.calibrated = True

        print("Stationary bias calibration completed.")
        return True

    def calibrate_gravity_alignment(self, measurements: list[list[Any]]) -> bool:
        """
        Compute rotation matrix to align with gravity.

        Args:
            measurements: Acceleration measurements

        Returns:
            Boolean indicating successful alignment
        """
        try:
            # Compute normalized gravity vector
            g_normalized, gravity_mag = self.compute_trimmed_mean(measurements)

            # Compute rotation matrix
            R_matrix = self.compute_rotation_matrix(g_normalized)

            # Store results
            self.gravity_rotation = Rotation.from_matrix(matrix=R_matrix)
            self.gravity_magnitude = gravity_mag

            # Update data processor
            self.data_processor.R_gravity = self.gravity_rotation
            self.data_processor.gravity_mag = self.gravity_magnitude

            print("Gravity alignment calibration completed.")
            return True

        except Exception as e:
            print(f"Gravity alignment failed: {e}")
            return False

    def detect_driving_motion(self) -> bool:
        """
        Detect when the vehicle starts moving.

        Returns:
            Boolean indicating movement detected
        """
        not_stationary_count = 0
        print("Waiting for vehicle to start moving...")

        while not_stationary_count < self.movement_detection_threshold:
            data = self.serial_reader.read_data()
            if data is None:
                continue

            ax, ay, az, gx, gy, gz, dt = data

            stationary = self.data_processor.is_stationary([ax, ay, az], [gx, gy, gz])

            if not stationary:
                not_stationary_count += 1
            else:
                not_stationary_count = 0

        print("Vehicle movement detected.")
        return True

    def collect_driving_data(self) -> Optional[list[list[Any]]]:
        """
        Collect acceleration data during continuous vehicle movement.

        Returns:
            List of acceleration measurements or None
        """
        driving_accel_data = []
        samples_collected = 0

        print(f"Collecting {self.driving_samples} driving motion samples...")

        while samples_collected < self.driving_samples:
            data = self.serial_reader.read_data()
            if data is None:
                continue

            ax, ay, az, gx, gy, gz, dt = data

            # Ensure we're still moving
            stationary = self.data_processor.is_stationary([ax, ay, az], [gx, gy, gz])
            if stationary:
                print("Vehicle stopped moving during data collection.")
                return None

            driving_accel_data.append([ax, ay, az])
            samples_collected += 1

        return driving_accel_data

    def calibrate_driving_direction(self, driving_data: list[list[Any]]) -> bool:
        """
        Compute rotation matrix for driving direction alignment.

        Args:
            driving_data: Acceleration measurements during driving

        Returns:
            Boolean indicating successful calibration
        """
        try:
            # Compute mean acceleration during driving
            mean_accel = np.mean(driving_data, axis=0)

            # Project acceleration to horizontal plane
            horizontal_accel = mean_accel.copy()
            horizontal_accel[2] = 0

            # Compute driving direction rotation
            driving_rotation_matrix = self.compute_rotation_matrix(horizontal_accel)

            # Store results
            self.driving_rotation = Rotation.from_matrix(driving_rotation_matrix)

            print("Driving direction calibration completed.")
            return True

        except Exception as e:
            print(f"Driving direction calibration failed: {e}")
            return False

    def perform_full_calibration(self, stationary_measurements: list[list[Any]]) -> bool:
        """
        Perform comprehensive sensor calibration.

        Args:
            stationary_measurements: Acceleration measurements during stationary period

        Returns:
            Boolean indicating successful full calibration
        """
        # Stationary bias calibration
        if not self.calibrate_stationary_bias():
            print("Stationary bias calibration failed.")
            return False

        # Gravity alignment
        if not self.calibrate_gravity_alignment(stationary_measurements):
            print("Gravity alignment calibration failed.")
            return False

        # Wait for and detect driving motion
        if not self.detect_driving_motion():
            return False

        # Collect driving data
        driving_data = self.collect_driving_data()
        if driving_data is None:
            return False

        # Calibrate driving direction
        return self.calibrate_driving_direction(driving_data)
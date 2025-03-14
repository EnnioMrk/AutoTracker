import numpy as np
import filterpy.kalman as kf
from scipy.spatial.transform import Rotation
import time


class EKF:
    def __init__(self, dt=0.01, process_noise=0.01, accel_noise=0.05, gyro_noise=1, initial_gyro_bias=None):
        self.n_states = 7  # [q0, q1, q2, q3, wx_bias, wy_bias, wz_bias]
        self.n_measurements = 3
        self.ekf = kf.ExtendedKalmanFilter(dim_x=self.n_states, dim_z=self.n_measurements)

        # Initialize state with provided biases or zeros
        if initial_gyro_bias is not None:
            self.ekf.x = np.array([1.0, 0.0, 0.0, 0.0,
                                   initial_gyro_bias[0], initial_gyro_bias[1], initial_gyro_bias[2]])
        else:
            self.ekf.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Covariance matrix tuning
        self.ekf.P = np.eye(self.n_states) * 0.1
        self.ekf.P[0:4, 0:4] *= 0.01
        self.ekf.P[4:7, 4:7] *= 1.0

        # Process noise configuration
        self.Q = np.eye(self.n_states) * process_noise
        self.Q[0:4, 0:4] *= 0.1
        self.Q[4:7, 4:7] = np.eye(3) * (gyro_noise ** 2)
        self.ekf.Q = self.Q

        # Measurement noise
        self.ekf.R = np.eye(self.n_measurements) * (accel_noise ** 2)
        self.min_accel_noise = accel_noise
        self.max_accel_noise = accel_noise * 10

        # System parameters
        self.dt = dt
        self.gyro_noise = gyro_noise
        self.gravity_mag = 9.81
        self.last_time = time.time()

        # Bias estimation
        self.bias_window_size = 100
        self.bias_history = []
        self.stationary_threshold = 0.05  # rad/s
        self.initialized = False

    def reset(self, initial_gyro_bias=None):
        if initial_gyro_bias is not None:
            self.ekf.x = np.array([1.0, 0.0, 0.0, 0.0,
                                   initial_gyro_bias[0], initial_gyro_bias[1], initial_gyro_bias[2]])
        else:
            self.ekf.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ekf.P = np.eye(self.n_states) * 0.1
        self.bias_history = []
        self.initialized = False

    def normalize_quaternion(self):
        q = self.ekf.x[0:4]
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            self.ekf.x[0:4] = q / q_norm

    def quaternion_to_rotation(self, q):
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    def quaternion_multiply(self, q1, q2):
        a1, b1, c1, d1 = q1
        a2, b2, c2, d2 = q2
        return np.array([
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
        ])

    def state_transition_function(self, x, dt, gyro_data):
        q = x[0:4]
        bias = x[4:7]
        w_corrected = gyro_data - bias
        w_quat = np.array([0, w_corrected[0], w_corrected[1], w_corrected[2]])
        q_dot = 0.5 * self.quaternion_multiply(q, w_quat)
        q_new = q + q_dot * dt
        return np.concatenate([q_new / np.linalg.norm(q_new), bias])

    def jacobian_function(self, x, dt, gyro_data):
        q = x[0:4]
        bias = x[4:7]
        w_corrected = gyro_data - bias
        F = np.eye(self.n_states)
        Omega = np.array([
            [0, -w_corrected[0], -w_corrected[1], -w_corrected[2]],
            [w_corrected[0], 0, w_corrected[2], -w_corrected[1]],
            [w_corrected[1], -w_corrected[2], 0, w_corrected[0]],
            [w_corrected[2], w_corrected[1], -w_corrected[0], 0]
        ])
        F[0:4, 0:4] += 0.5 * dt * Omega
        F[0:4, 4] = 0.5 * dt * np.array([q[1], -q[0], -q[3], q[2]])
        F[0:4, 5] = 0.5 * dt * np.array([q[2], q[3], -q[0], -q[1]])
        F[0:4, 6] = 0.5 * dt * np.array([q[3], -q[2], q[1], -q[0]])
        return F

    def measurement_function(self, x):
        q = x[0:4]
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T @ np.array([0, 0, self.gravity_mag])

    def measurement_jacobian(self, x):
        q = x[0:4]
        H = np.zeros((3, 7))
        q0, q1, q2, q3 = q
        g = self.gravity_mag

        H[0, 0] = 2 * g * (q1 * q3 - q0 * q2)
        H[0, 1] = 2 * g * (q0 * q3 + q1 * q2)
        H[0, 2] = 2 * g * (q2 * q3 - q0 * q1)
        H[0, 3] = 2 * g * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)

        H[1, 0] = 2 * g * (q2 * q3 + q0 * q1)
        H[1, 1] = 2 * g * (q1 * q3 - q0 * q2)
        H[1, 2] = 2 * g * (q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2)
        H[1, 3] = 2 * g * (q0 * q3 - q1 * q2)

        H[2, 0] = 2 * g * (q0 * q3 - q1 * q2)
        H[2, 1] = 2 * g * (q0 * q2 + q1 * q3)
        H[2, 2] = 2 * g * (q1 ** 2 + q3 ** 2 - q0 ** 2 - q2 ** 2)
        H[2, 3] = 2 * g * (q2 * q3 - q0 * q1)
        return H

    def _calculate_gravity_alignment(self, accel_data):
        """Calculate initial orientation from accelerometer data."""
        accel_norm = np.linalg.norm(accel_data)
        if accel_norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

        # Normalize accelerometer measurement
        g_dir = accel_data / accel_norm
        target_gravity = np.array([0, 0, 1])  # Earth frame gravity direction

        # Calculate rotation axis and angle
        axis = np.cross(g_dir, target_gravity)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            if np.dot(g_dir, target_gravity) > 0:
                return np.array([1.0, 0.0, 0.0, 0.0])  # No rotation needed
            else:
                return np.array([0.0, 1.0, 0.0, 0.0])  # 180° flip

        axis /= axis_norm
        angle = np.arccos(np.clip(np.dot(g_dir, target_gravity), -1.0, 1.0))

        # Create quaternion (w, x, y, z)
        return np.array([
            np.cos(angle / 2),
            axis[0] * np.sin(angle / 2),
            axis[1] * np.sin(angle / 2),
            axis[2] * np.sin(angle / 2)
        ])

    def set_initial_orientation(self, accel_data):
        # Add low-pass filtering and magnitude validation
        accel_norm = np.linalg.norm(accel_data)

        # Validate gravity magnitude (should be ~1g ±0.2g)
        if not (0.8 * self.gravity_mag < accel_norm < 1.2 * self.gravity_mag):
            return False

        # Low-pass filter for initial alignment
        alpha = 0.98  # Weight for complementary filter
        self.ekf.x[0:4] = self._calculate_gravity_alignment(accel_data)

        # Initialize covariance properly
        self.ekf.P[0:4, 0:4] = np.eye(4) * 0.01  # Low initial orientation uncertainty
        self.initialized = True
        return True

    def update_bias_with_moving_average(self, gyro):
        gyro_norm = np.linalg.norm(gyro)

        if gyro_norm < self.stationary_threshold:
            self.bias_history.append(gyro.copy())
            if len(self.bias_history) > self.bias_window_size:
                self.bias_history.pop(0)

            if len(self.bias_history) >= 10:
                # Use weighted average with higher weight for recent samples
                weights = np.linspace(0.1, 1.0, len(self.bias_history))
                weights /= weights.sum()
                new_bias = np.average(self.bias_history, axis=0, weights=weights)

                # Adaptive learning rate based on stationarity duration
                alpha = min(0.1, 1.0 / len(self.bias_history))
                self.ekf.x[4:7] = (1 - alpha) * self.ekf.x[4:7] + alpha * new_bias

    def adaptive_measurement_noise(self, accel):
        gravity_deviation = abs(np.linalg.norm(accel) - self.gravity_mag) / self.gravity_mag
        noise_scale = np.clip(gravity_deviation * 10, 1.0, 10.0)
        self.ekf.R = np.eye(3) * (self.min_accel_noise * noise_scale) ** 2
        return noise_scale

    def update(self, accel, gyro, dt=None):
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
        if dt <= 0:
            dt = self.dt

        if not self.initialized:
            self.set_initial_orientation(accel)

        self.update_bias_with_moving_average(gyro)
        self.adaptive_measurement_noise(accel)

        # Prediction step
        self.ekf.x = self.state_transition_function(self.ekf.x, dt, gyro)
        F = self.jacobian_function(self.ekf.x, dt, gyro)
        self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q
        self.normalize_quaternion()

        # Measurement update
        predicted_accel = self.measurement_function(self.ekf.x)
        accel_mag = np.linalg.norm(accel)
        innovation = np.linalg.norm(accel - predicted_accel)

        gravity_threshold = 0.2 * self.gravity_mag  # ~2 m/s²
        innovation_threshold = 0.3 * self.gravity_mag  # ~3 m/s²

        if (abs(accel_mag - self.gravity_mag) < gravity_threshold and
                innovation < innovation_threshold):
            self.ekf.update(accel, self.measurement_jacobian, self.measurement_function)
            self.normalize_quaternion()

        return self.ekf.x

    def get_rotation(self):
        return Rotation.from_quat(self.ekf.x[[1, 2, 3, 0]])


    def get_euler(self):
        return self.get_rotation().as_euler('xyz')

    def get_quaternion(self):
        return self.ekf.x[0:4]

    def get_bias(self):
        return self.ekf.x[4:7]

    def get_linear_acceleration(self, accel):
        return accel - self.measurement_function(self.ekf.x)
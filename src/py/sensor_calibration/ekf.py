import numpy as np
import filterpy.kalman as kf
from scipy.spatial.transform import Rotation
import time


class EKF:
    def __init__(self, dt=0.01, process_noise=0.01, accel_noise=0.05, gyro_noise=1):
        """
        Initialize Extended Kalman Filter for IMU orientation tracking.

        Parameters:
        -----------
        dt : float
            Time step between measurements (default: 0.01s)
        process_noise : float
            Base process noise covariance (default: 0.01) - applied to all states initially
        accel_noise : float
            Accelerometer measurement noise (default: 0.05) in m/s² (when sensor values are converted)
        gyro_noise : float
            Gyroscope process noise (default: 1) in rad/s, for gyro bias drift
        """
        # State vector [q0, q1, q2, q3, wx_bias, wy_bias, wz_bias]
        self.n_states = 7
        self.n_measurements = 3  # 3D accelerometer readings

        self.ekf = kf.ExtendedKalmanFilter(dim_x=self.n_states, dim_z=self.n_measurements)

        # Initialize state with identity quaternion and zero biases
        self.ekf.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Initialize state covariance matrix
        self.ekf.P = np.eye(self.n_states) * 0.1

        # Process noise covariance matrix
        self.Q = np.eye(self.n_states) * process_noise
        self.Q[0:4, 0:4] *= 0.1  # lower noise for quaternion dynamics
        self.Q[4:7, 4:7] = np.eye(3) * (gyro_noise ** 2)  # gyro bias noise (now in rad/s)
        self.ekf.Q = self.Q

        # Measurement noise covariance matrix
        self.ekf.R = np.eye(self.n_measurements) * (accel_noise ** 2)  # accel noise variance in (m/s²)²

        self.dt = dt
        self.gyro_noise = gyro_noise
        self.gravity_mag = 9.81  # gravity in m/s²
        self.last_time = time.time()

    def normalize_quaternion(self):
        q = self.ekf.x[0:4]
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            self.ekf.x[0:4] = q / q_norm

    def quaternion_to_rotation(self, q):
        # Note: scipy Rotation expects quaternion in [x, y, z, w]
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    def quaternion_multiply(self, q1, q2):
        a1, b1, c1, d1 = q1
        a2, b2, c2, d2 = q2
        return np.array([
            a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2
        ])

    def state_transition_function(self, x, dt, gyro_data):
        q = x[0:4]
        bias = x[4:7]
        # gyro_data should be in rad/s now
        w_corrected = gyro_data - bias
        w_quat = np.array([0, w_corrected[0], w_corrected[1], w_corrected[2]])
        q_dot = 0.5 * self.quaternion_multiply(q, w_quat)
        q_new = q + q_dot * dt
        q_new = q_new / np.linalg.norm(q_new)
        x_new = np.concatenate([q_new, bias])
        return x_new

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
        R = self.quaternion_to_rotation(q)
        gravity = np.array([0, 0, self.gravity_mag])
        expected_accel = R.T @ gravity
        return expected_accel

    def measurement_jacobian(self, x):
        q = x[0:4]
        H = np.zeros((self.n_measurements, self.n_states))
        q0, q1, q2, q3 = q
        g = self.gravity_mag
        H[0, 0] = 2 * g * (q1*q3 - q0*q2)
        H[0, 1] = 2 * g * (q0*q3 + q1*q2)
        H[0, 2] = 2 * g * (q2*q3 - q0*q1)
        H[0, 3] = 2 * g * (q0*q0 - q1*q1 - q2*q2 + q3*q3)

        H[1, 0] = 2 * g * (q2*q3 + q0*q1)
        H[1, 1] = 2 * g * (q1*q3 - q0*q2)
        H[1, 2] = 2 * g * (q0*q0 + q1*q1 - q2*q2 - q3*q3)
        H[1, 3] = 2 * g * (q0*q3 - q1*q2)

        H[2, 0] = 2 * g * (q0*q3 - q1*q2)
        H[2, 1] = 2 * g * (q0*q2 + q1*q3)
        H[2, 2] = 2 * g * (q1*q1 + q3*q3 - q0*q0 - q2*q2)
        H[2, 3] = 2 * g * (q2*q3 - q0*q1)
        return H

    def update(self, accel, gyro, dt=None):
        """
        Update the filter with new measurements. The quaternion integration is done
        using an axis-angle method, and the accelerometer update is gated based on how
        close the measured acceleration is to the predicted gravity vector.
        """
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
        if dt <= 0:
            dt = self.dt

        # ----------------------------
        # Improved Prediction Step
        # ----------------------------
        # Extract current quaternion and bias.
        x = self.ekf.x
        q = x[0:4]
        bias = x[4:7]
        # Correct gyro measurements (gyro now in rad/s).
        w_corrected = gyro - bias
        w_norm = np.linalg.norm(w_corrected)

        # Use axis-angle based update if angular rate is significant.
        if w_norm > 1e-6:
            theta = w_norm * dt
            # Compute quaternion increment using axis-angle representation.
            dq = np.hstack([np.cos(theta / 2), np.sin(theta / 2) * (w_corrected / w_norm)])
            # Apply the rotation increment.
            q_new = self.quaternion_multiply(q, dq)
            q_new /= np.linalg.norm(q_new)
        else:
            q_new = q

        # The bias remains unchanged.
        x_pred = np.concatenate([q_new, bias])

        # Compute the Jacobian for the prediction step.
        F = self.jacobian_function(self.ekf.x, dt, gyro)
        self.ekf.x = x_pred
        self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q
        self.normalize_quaternion()

        # ----------------------------
        # Improved Measurement Update
        # ----------------------------
        # Compute the predicted accelerometer reading (should be close to [0, 0, g] in sensor frame)
        predicted_accel = self.measurement_function(self.ekf.x)

        # Only update if the measured acceleration is close to the expected gravity vector.
        # This threshold (here 0.5 m/s²) may be tuned based on your sensor noise and dynamics.
        if np.linalg.norm(accel - predicted_accel) < 0.5:
            self.ekf.update(z=accel,
                            HJacobian=self.measurement_jacobian,
                            Hx=self.measurement_function)
            self.normalize_quaternion()

        return self.ekf.x

    def get_rotation(self):
        q = self.ekf.x[0:4]
        return Rotation.from_quat([q[1], q[2], q[3], q[0]])

    def get_euler_angles(self):
        return self.get_rotation().as_euler('xyz')

    def get_quaternion(self):
        return self.ekf.x[0:4]

    def get_gyro_bias(self):
        return self.ekf.x[4:7]

    def set_gravity_magnitude(self, gravity_mag):
        self.gravity_mag = gravity_mag

    def get_linear_accel(self, accel):
        R = self.quaternion_to_rotation(self.ekf.x[0:4])
        gravity_sensor = R.T @ np.array([0, 0, self.gravity_mag])
        return accel - gravity_sensor
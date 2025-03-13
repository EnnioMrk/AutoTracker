import numpy as np
from filterpy.kalman import ExtendedKalmanFilter, KalmanFilter
from scipy.spatial.transform import Rotation
from typing import Any, Optional, Sequence, Tuple


# --- Quaternion RK4 Integration ---
def quaternion_rk4_integration(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Runge-Kutta 4th order integration for quaternion update.

    Args:
        q: Current quaternion [w, x, y, z]
        omega: Angular velocity vector [0, wx, wy, wz]
        dt: Time step

    Returns:
        Updated quaternion [w, x, y, z]
    """

    def qdot(q_val, omega_val):
        return 0.5 * quaternion_multiply_np(q_val, omega_val)

    k1 = qdot(q, omega)
    k2 = qdot(q + dt / 2.0 * k1, omega)
    k3 = qdot(q + dt / 2.0 * k2, omega)
    k4 = qdot(q + dt * k3, omega)

    q_next = q + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return q_next / np.linalg.norm(q_next)  # Normalize the quaternion


def quaternion_multiply_np(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions using numpy arrays.
    Both q and r are assumed to be in the format [w, x, y, z].
    """
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


# --- Extended Kalman Filter (EKF) ---
class EKF(ExtendedKalmanFilter):  # Renamed to avoid confusion with ESKF
    """
    Extended Kalman Filter for IMU sensor fusion.
    Modified to work with acceleration in g units (1g = 9.81 m/s²)
    """

    def __init__(self, dt: float, process_noise: float = 0.01, measurement_noise: float = 0.1,
                 quaternion_process_noise: float = 0.001, gyro_noise: float = 0.01):
        super().__init__(dim_x=10, dim_z=6)
        self.x = np.zeros(10)
        self.x[6] = 1.0
        self.Q = np.eye(10) * process_noise
        self.Q[6:10, 6:10] *= quaternion_process_noise
        self.R = np.eye(6) * measurement_noise
        self.R[3:6, 3:6] *= gyro_noise
        self.P = np.eye(10) * 100
        self.P[6:10, 6:10] = np.eye(4) * 0.1
        self.dt = dt
        self.gravity_mag = 1.0  # Now in g units (1.0g)
        self.g = np.array([0, 0, self.gravity_mag])  # g vector in g units
        self.g_mps2 = 9.81  # Conversion factor from g to m/s²
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

    def set_gravity_magnitude(self, gravity_mag: float) -> None:
        """Set gravity magnitude in g units (1.0 = 1g = 9.81 m/s²)"""
        self.gravity_mag = gravity_mag
        self.g = np.array([0, 0, self.gravity_mag])

    def set_g_conversion_factor(self, g_mps2: float) -> None:
        """Set the conversion factor from g to m/s² (default is 9.81)"""
        self.g_mps2 = g_mps2

    def set_biases(self, accel_bias: Sequence[float], gyro_bias: Sequence[float]) -> None:
        self.accel_bias = np.array(accel_bias)
        self.gyro_bias = np.array(gyro_bias)

    def set_initial_orientation(self, quaternion: Sequence[float]) -> None:
        self.x[6:10] = quaternion

    def predict_x(self, x: np.ndarray, dt: float, u: Optional[Sequence[float]] = None, **kwargs: Any) -> np.ndarray:
        pos = x[0:3]
        vel = x[3:6]
        q = x[6:10]

        if u is None or len(u) < 6:
            ax_b, ay_b, az_b = 0.0, 0.0, 0.0
            wx, wy, wz = 0.0, 0.0, 0.0
        else:
            u = list(u)
            ax_b, ay_b, az_b = u[0:3]
            wx, wy, wz = u[3:6]

        # Convert acceleration from g units to m/s²
        accel_body_g = np.array([ax_b, ay_b, az_b])
        accel_body_mps2 = accel_body_g * self.g_mps2

        quat_converted = [q[1], q[2], q[3], q[0]]
        r = Rotation.from_quat(quat=quat_converted)

        # Convert gravity from g to m/s² for physics calculations
        g_mps2 = self.g * self.g_mps2

        # Apply rotation and add gravity (in m/s²)
        accel_world_mps2 = r.apply(accel_body_mps2) + g_mps2

        new_pos = pos + vel * dt + 0.5 * accel_world_mps2 * dt ** 2
        new_vel = vel + accel_world_mps2 * dt

        # Quaternion update using RK4
        omega = np.array([0.0, wx, wy, wz])
        new_q = quaternion_rk4_integration(q, omega, dt)

        x_new = np.zeros(10)
        x_new[0:3] = new_pos
        x_new[3:6] = new_vel
        x_new[6:10] = new_q
        return x_new

    def predict(self, u: Optional[Sequence[float]] = None, B: Any = None, F: Any = None,
                Q: Optional[np.ndarray] = None, dt: Optional[float] = None) -> None:
        if dt is None:
            dt = self.dt
        self.x = self.predict_x(self.x, dt, u)
        F_jac = self._compute_F_jacobian(self.x, dt, u)
        if Q is None:
            Q = self.Q
        self.P = F_jac @ self.P @ F_jac.T + Q

    def _compute_F_jacobian(self, x: np.ndarray, dt: float, u: Optional[Sequence[float]] = None) -> np.ndarray:
        F = np.eye(10)
        F[0:3, 3:6] = np.eye(3) * dt
        q = x[6:10]
        if u is not None and len(u) >= 6:
            u = list(u)
            wx, wy, wz = u[3:6]
            F[6:10, 6:10] = np.eye(4) + 0.5 * dt * np.array(
                [  # Still using first order jacobian, RK4 improves prediction but not jacobian.
                    [0, -wx, -wy, -wz],
                    [wx, 0, wz, -wy],
                    [wy, -wz, 0, wx],
                    [wz, wy, -wx, 0]
                ])
        return F

    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function (accelerometer and gyroscope readings).
        Returns accelerometer readings in g units.
        """
        q = x[6:10]
        quat_converted = [q[1], q[2], q[3], q[0]]
        r = Rotation.from_quat(quat=quat_converted)

        # For measurement model, use g units (not m/s²)
        accel_world = -self.g  # Expected acceleration in g units when stationary
        accel_meas = r.apply(accel_world, inverse=True)
        gyro_meas = np.zeros(3)
        return np.concatenate([accel_meas, gyro_meas])

    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((6, 10))
        q = x[6:10]
        H[0:3, 6:10] = self._compute_h_quaternion_jacobian(q)
        return H

    def _compute_h_quaternion_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian of measurement model with respect to quaternion.
        Uses gravity in g units.
        """
        w, x_val, y_val, z_val = q
        g = self.gravity_mag  # Use g units for the Jacobian
        return np.array([
            [2 * g * x_val, 2 * g * w, -2 * g * z_val, 2 * g * y_val],
            [2 * g * y_val, 2 * g * z_val, 2 * g * w, -2 * g * x_val],
            [2 * g * z_val, -2 * g * y_val, 2 * g * x_val, 2 * g * w]
        ])

    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None, H: Any = None,
               HJacobian: Any = None, Hx: Any = None, args: Tuple = (), hx_args: Tuple = (),
               residual: Any = np.subtract) -> None:
        def HJacobian_fn(x: np.ndarray, *args: Any) -> np.ndarray:
            return self.H_jacobian(x)

        def Hx_fn(x: np.ndarray, *args: Any) -> np.ndarray:
            return self.h(x)

        if R is None:
            R = self.R
        super().update(z, R=R, HJacobian=HJacobian_fn, Hx=Hx_fn, args=args, hx_args=hx_args, residual=residual)
        q_norm = np.linalg.norm(self.x[6:10])
        if q_norm > 0:
            self.x[6:10] /= q_norm

    def get_position(self) -> np.ndarray:
        return self.x[0:3]

    def get_velocity(self) -> np.ndarray:
        return self.x[3:6]

    def get_quaternion(self) -> np.ndarray:
        return self.x[6:10]

    def get_rotation_matrix(self) -> np.ndarray:
        q = self.x[6:10]
        r = Rotation.from_quat(quat=[q[1], q[2], q[3], q[0]])
        return r.as_matrix()

    def get_euler_angles(self) -> np.ndarray:
        q = self.x[6:10]
        r = Rotation.from_quat(quat=[q[1], q[2], q[3], q[0]])
        return r.as_euler('xyz')

    def get_linear_acceleration(self, corrected_accel_body: np.ndarray) -> np.ndarray:
        """
        Compute linear acceleration in world frame (gravity removed).

        Args:
            corrected_accel_body: Accelerometer data after bias correction in body frame (in g units)

        Returns:
            Linear acceleration in world frame [ax, ay, az] (in g units)
        """
        q = self.get_quaternion()
        rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # [x, y, z, w] for SciPy

        # Transform accelerometer readings from body to world frame (still in g units)
        accel_world = rotation.apply(corrected_accel_body)

        # Remove gravity effect (subtract g vector in g units)
        linear_accel = accel_world - self.g

        return linear_accel


# --- Error State Kalman Filter (ESKF) ---
class ESKF(KalmanFilter):
    """
    Error State Kalman Filter for IMU sensor fusion.
    Modified to work with acceleration in g units (1g = 9.81 m/s²)
    """

    def __init__(self, dt: float, process_noise: float = 1e-5, measurement_noise: float = 0.1,
                 gyro_noise: float = 0.01, gyro_bias_process_noise: float = 1e-8,
                 accel_noise: float = 0.1, accel_bias_process_noise: float = 1e-7):
        super().__init__(dim_x=15, dim_z=6)  # 15 error states, 6 measurements

        # State Vector: Error states (delta_orientation, delta_velocity, delta_position, gyro_bias, accel_bias)
        # x = [d_theta_x, d_theta_y, d_theta_z,  dv_x, dv_y, dv_z,  dp_x, dp_y, dp_z,  bg_x, bg_y, bg_z,  ba_x, ba_y, ba_z]

        self.dt = dt
        self.gravity_mag = 1.0  # Now in g units (1.0g)
        self.g_vec = np.array([0, 0, self.gravity_mag])  # g vector in g units
        self.g_mps2 = 9.81  # Conversion factor from g to m/s²

        # Nominal States (these are estimated outside the ESKF, using direct integration)
        self.nominal_position = np.zeros(3)
        self.nominal_velocity = np.zeros(3)
        self.nominal_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

        # Initial Error State and Covariance (initially zero error and small covariance)
        self.x = np.zeros(15)  # Error state initialized to zero
        self.P = np.eye(15) * 0.01  # Small initial covariance for error states

        # Process Noise Covariance Matrix Q (Error state propagation noise)
        self.Q = np.diag([
            process_noise, process_noise, process_noise,  # Orientation error
            process_noise, process_noise, process_noise,  # Velocity error
            process_noise, process_noise, process_noise,  # Position error
            gyro_bias_process_noise, gyro_bias_process_noise, gyro_bias_process_noise,  # Gyro bias
            accel_bias_process_noise, accel_bias_process_noise, accel_bias_process_noise  # Accel bias
        ])

        # Measurement Noise Covariance Matrix R
        self.R = np.diag([
            accel_noise, accel_noise, accel_noise,  # Accelerometer noise
            gyro_noise, gyro_noise, gyro_noise  # Gyro noise
        ])

        # Measurement Matrix H (Observation model Jacobian - relates error state to measurement)
        self.H = np.zeros((6, 15))
        self.H[0:3, 6:9] = np.eye(3)  # Position error affects accel measurement (indirectly through gravity)
        self.H[0:3, 12:15] = -np.eye(3)  # Accel bias directly affects accel measurement
        self.H[3:6, 9:12] = -np.eye(3)  # Gyro bias directly affects gyro measurement
        self.H[3:6, 0:3] = -np.eye(3)  # Orientation error affects gyro measurement

        # State Transition Matrix F (Jacobian of error state propagation) - Will be computed in predict step

    def set_gravity_magnitude(self, gravity_mag: float) -> None:
        """Set gravity magnitude in g units (1.0 = 1g = 9.81 m/s²)"""
        self.gravity_mag = gravity_mag
        self.g_vec = np.array([0, 0, self.gravity_mag])

    def set_g_conversion_factor(self, g_mps2: float) -> None:
        """Set the conversion factor from g to m/s² (default is 9.81)"""
        self.g_mps2 = g_mps2

    def set_biases(self, accel_bias: Sequence[float], gyro_bias: Sequence[float]) -> None:
        self.accel_bias = np.array(accel_bias)
        self.gyro_bias = np.array(gyro_bias)
        self.nominal_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Reset orientation on bias set
        self.nominal_velocity = np.zeros(3)
        self.nominal_position = np.zeros(3)

    def set_initial_orientation(self, quaternion: Sequence[float]) -> None:
        self.nominal_quaternion = quaternion

    def predict(self, u: Optional[Sequence[float]] = None, dt: Optional[float] = None, Q: Optional[np.ndarray] = None,
                **kwargs: Any) -> None:
        if dt is None:
            dt = self.dt
        if u is None or len(u) < 6:
            accel_meas = np.zeros(3)
            gyro_meas = np.zeros(3)
        else:
            accel_meas = np.array(u[0:3])  # Acceleration in g units
            gyro_meas = np.array(u[3:6])

        # 1. Nominal State Propagation (using measurements and previous nominal states)
        omega_nominal_body = gyro_meas - self.gyro_bias  # Correct gyro measurements for bias
        accel_nominal_body = accel_meas - self.accel_bias  # Correct accel measurements for bias (in g units)

        # Convert acceleration from g to m/s² for physics calculations
        accel_nominal_body_mps2 = accel_nominal_body * self.g_mps2

        # Quaternion Propagation (RK4 for nominal quaternion)
        omega_vec = np.array([0.0, omega_nominal_body[0], omega_nominal_body[1], omega_nominal_body[2]])
        self.nominal_quaternion = quaternion_rk4_integration(self.nominal_quaternion, omega_vec, dt)

        # Velocity Propagation (in world frame)
        quat_nominal_conj = self.conjugate_quaternion(self.nominal_quaternion)
        rotation_body_to_world = Rotation.from_quat(
            [self.nominal_quaternion[1], self.nominal_quaternion[2], self.nominal_quaternion[3],
             self.nominal_quaternion[0]]).as_matrix()  # [x,y,z,w] for scipy

        # Convert gravity from g to m/s² for physics calculations
        g_vec_mps2 = self.g_vec * self.g_mps2

        # Apply rotation and add gravity (in m/s²)
        accel_world_mps2 = rotation_body_to_world @ accel_nominal_body_mps2 + g_vec_mps2
        self.nominal_velocity += accel_world_mps2 * dt

        # Position Propagation (in world frame)
        self.nominal_position += self.nominal_velocity * dt

        # 2. Error State Propagation (Error states propagate according to linearized dynamics)
        # Compute State Transition Matrix F (Jacobian of error state propagation)
        F = self._compute_F_jacobian(omega_nominal_body, accel_nominal_body_mps2, rotation_body_to_world, dt)
        self.F = F  # Store for update step (though filterpy might not need this stored)

        # Propagate error covariance matrix
        if Q is None:
            Q = self.Q
        self.P = F @ self.P @ F.T + Q

        # Error state x propagation is assumed to be x = F @ x, but since x is initialized to zero and no control input on error state, x remains zero in predict.
        self.x = F @ self.x  # Technically should be here for full ESKF, but as x is error, and we reset in update, and no error input, it often stays zero.

    def _compute_F_jacobian(self, omega_nominal_body, accel_nominal_body_mps2, rotation_body_to_world, dt):
        """Computes the state transition Jacobian F for the ESKF."""
        F = np.eye(15)
        # Orientation Error to Velocity Error Jacobian (Eq 7.77, 7.80 in Solà)
        F[3:6, 0:3] = -skew_symmetric_matrix(rotation_body_to_world @ accel_nominal_body_mps2) * dt
        # Velocity Error to Position Error Jacobian
        F[6:9, 3:6] = np.eye(3) * dt
        # Orientation Error Propagation (Eq 7.79 in Solà, linearized quaternion kinematics)
        F[0:3, 0:3] = np.eye(3) - skew_symmetric_matrix(omega_nominal_body) * dt
        # Gyro Bias Error Propagation (Bias assumed to be random walk, so F_bg_bg = I) - already I due to initialization
        # Accel Bias Error Propagation (Bias assumed to be random walk, so F_ba_ba = I) - already I due to initialization

        return F

    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None, H: Any = None,
               HJacobian: Any = None, Hx: Any = None, args: Tuple = (), hx_args: Tuple = (),
               residual: Any = np.subtract) -> None:
        if R is None:
            R = self.R
        if H is None:
            H = self.H

        # 1. Innovation/Measurement residual
        y = z - self.h()  # Measurement residual is measurement minus predicted measurement (h() in ESKF is error-state predicted measurement)

        # 2. Kalman Gain Calculation
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 3. Error state update
        self.x = self.x + K @ y

        # 4. Covariance update
        self.P = (np.eye(15) - K @ H) @ self.P

        # 5. Correct Nominal states using error states and reset error states to zero
        # Orientation correction (quaternion composition - Eq 7.104, 7.105 Solà)
        delta_orientation_quat = self.delta_quaternion_from_rotation_vector(
            self.x[0:3])  # Convert rotation vector error to quaternion
        self.nominal_quaternion = quaternion_multiply_np(self.nominal_quaternion,
                                                         delta_orientation_quat)  # Apply correction to nominal quaternion
        self.nominal_quaternion /= np.linalg.norm(self.nominal_quaternion)  # Normalize

        # Velocity and Position correction (additive error)
        self.nominal_velocity += self.x[3:6]
        self.nominal_position += self.x[6:9]

        # Bias correction (additive error)
        self.gyro_bias += self.x[9:12]
        self.accel_bias += self.x[12:15]

        # Reset error states to zero after correction (important for ESKF)
        self.x = np.zeros(15)

    def h(self) -> np.ndarray:
        """Measurement function for ESKF - predicts measurements based on error state (deviation from nominal).

        In ESKF, h(x) is the *predicted measurement error* given the error state x.
        However, for simplicity and using filterpy's structure, we are using it to return the *predicted measurement* itself.
        Then the residual is calculated as z - h().

        For IMU, the predicted measurement error is related to:
        - Position error (indirectly, affects gravity component in accel readings)
        - Accelerometer bias error
        - Gyro bias error
        - Orientation error (affects gyro readings)

        Here, we simplify and assume predicted measurement is just zero, and H matrix captures the sensitivity.
        So, h() returns zeros, and the H matrix is used in update step to relate error state to measurement residual.
        """
        return np.zeros(6)  # Predicted measurement error is assumed zero when error state is zero.

    def delta_quaternion_from_rotation_vector(self, delta_theta: np.ndarray) -> np.ndarray:
        """Converts a rotation vector (error orientation) to a delta quaternion."""
        norm_delta_theta = np.linalg.norm(delta_theta)
        if norm_delta_theta < 1e-8:
            delta_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        else:
            axis = delta_theta / norm_delta_theta
            angle = norm_delta_theta
            delta_quat = np.concatenate(([np.cos(angle / 2.0)], axis * np.sin(angle / 2.0)))  # [w, x, y, z]
        return delta_quat

    def conjugate_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Returns the conjugate of a quaternion."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def get_position(self) -> np.ndarray:
        return self.nominal_position

    def get_velocity(self) -> np.ndarray:
        return self.nominal_velocity

    def get_quaternion(self) -> np.ndarray:
        return self.nominal_quaternion

    def get_gyro_bias(self) -> np.ndarray:
        return self.gyro_bias

    def get_accel_bias(self) -> np.ndarray:
        return self.accel_bias

    def get_rotation_matrix(self) -> np.ndarray:
        r = Rotation.from_quat([self.nominal_quaternion[1], self.nominal_quaternion[2], self.nominal_quaternion[3],
                                self.nominal_quaternion[0]])  # [x,y,z,w] for scipy
        return r.as_matrix()

    def get_euler_angles(self) -> np.ndarray:
        r = Rotation.from_quat([self.nominal_quaternion[1], self.nominal_quaternion[2], self.nominal_quaternion[3],
                                self.nominal_quaternion[0]])  # [x,y,z,w] for scipy
        return r.as_euler('xyz')

    def get_linear_acceleration(self, corrected_accel_body: np.ndarray) -> np.ndarray:
        """
        Compute linear acceleration in world frame (gravity removed).

        Args:
            corrected_accel_body: Accelerometer data after bias correction in body frame (in g units)

        Returns:
            Linear acceleration in world frame [ax, ay, az] (in g units)
        """
        rotation = self.get_rotation_matrix()

        # Transform accelerometer readings from body to world frame (still in g units)
        accel_world = rotation @ corrected_accel_body

        # Remove gravity effect (subtract g vector in g units)
        linear_accel = accel_world - self.g_vec

        return linear_accel

def skew_symmetric_matrix(v):
    """Returns the skew-symmetric matrix of a vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
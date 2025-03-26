import numpy as np

class ComplementaryFilter:
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0

    def update(self, accel, gyro, dt):
        # Compute roll and pitch from accelerometer (in radians)
        accel_roll = np.arctan2(accel[1], accel[2])
        accel_pitch = np.arctan2(-accel[0], np.sqrt(accel[1] ** 2 + accel[2] ** 2))

        # Convert gyro from °/s to rad/s and compute delta angles
        gyro_rad = np.radians(gyro)
        delta_roll = gyro_rad[0] * dt
        delta_pitch = gyro_rad[1] * dt
        delta_yaw = gyro_rad[2] * dt

        # Integrate gyro deltas with previous angles
        gyro_roll = self.prev_roll + delta_roll
        gyro_pitch = self.prev_pitch + delta_pitch
        gyro_yaw = self.prev_yaw + delta_yaw

        # Fuse with accelerometer angles using complementary filter
        roll = self.alpha * gyro_roll + (1 - self.alpha) * accel_roll
        pitch = self.alpha * gyro_pitch + (1 - self.alpha) * accel_pitch
        yaw = gyro_yaw  # No accelerometer correction for yaw

        self.prev_roll = roll
        self.prev_pitch = pitch
        self.prev_yaw = yaw

        return roll, pitch, yaw
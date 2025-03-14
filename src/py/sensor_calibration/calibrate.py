import numpy as np
from typing import List, Tuple, Any

from numpy import ndarray, dtype, float64, complex128, floating, complexfloating, inexact, number, timedelta64
from scipy.spatial.transform import Rotation

def calculate_avg_accel(measurements: List, trim_percentage: float = 0.1):
    """
    Compute the average raw acceleration vector and its magnitude using trimmed mean.
    """
    if not 0.0 <= trim_percentage <= 0.5:
        raise ValueError("trim_percentage must be between 0.0 and 0.5")

    # Convert measurements to numpy array
    vectors = np.array([m for m in measurements])
    num_measurements = vectors.shape[0]

    if num_measurements == 0:
        return np.zeros(3, dtype=np.float64), 0.0

    # Create trimmed vectors by component
    trimmed_x = np.sort(vectors[:, 0])[
                int(num_measurements * trim_percentage):int(num_measurements * (1 - trim_percentage))]
    trimmed_y = np.sort(vectors[:, 1])[
                int(num_measurements * trim_percentage):int(num_measurements * (1 - trim_percentage))]
    trimmed_z = np.sort(vectors[:, 2])[
                int(num_measurements * trim_percentage):int(num_measurements * (1 - trim_percentage))]

    # Calculate means
    avg_vec = np.array([np.mean(trimmed_x), np.mean(trimmed_y), np.mean(trimmed_z)])

    gravity_mag = np.linalg.norm(avg_vec)
    if gravity_mag > 0:
        g_normalized = avg_vec / gravity_mag
    else:
        g_normalized = np.array([0, 0, 1])  # Default to world up if no signal

    return g_normalized, gravity_mag


def compute_rotation_matrix(g_normalized: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix to align measured gravity with world up [0, 0, 1].
    """
    world_up = np.array([0, 0, 1], dtype=np.float64)
    v = np.cross(g_normalized, world_up)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-6:
        return np.identity(3)
    v = v / v_norm
    dot_val = np.dot(g_normalized, world_up)
    theta = np.arccos(np.clip(dot_val, -1.0, 1.0))
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    R = np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def calibrate_sensors(measurements):
    """
    Calibrate sensors using a set of raw accelerometer measurements.
    """
    g_normalized, gravity_mag = calculate_avg_accel(measurements)
    R_matrix = compute_rotation_matrix(g_normalized)
    R_calib = Rotation.from_matrix(matrix=R_matrix)
    return R_calib, gravity_mag # Return the Rotation object
import numpy as np
from typing import List, Tuple, Any

from numpy import ndarray, dtype, float64, complex128, floating, complexfloating, inexact, number, timedelta64
from scipy.spatial.transform import Rotation


class VectorMeasurement:
    def __init__(self, a: float, b: float, c: float):
        self.vec = np.array([a, b, c], dtype=np.float64)


def calculate_avg_accel(measurements: List[VectorMeasurement], trim_percentage: float = 0.1) -> tuple[ndarray[
    tuple[int], dtype[Any]], float] | tuple[float | ndarray[tuple[int, ...], dtype[float64]] | ndarray[
    tuple[int, ...], dtype[complex128]] | ndarray[tuple[int, ...], dtype[floating]] | ndarray[
                                                tuple[int, ...], dtype[complexfloating]] | ndarray[
                                                tuple[int, ...], dtype[inexact]] | ndarray[
                                                tuple[int, ...], dtype[number]] | ndarray[
                                                tuple[int, ...], dtype[timedelta64]] | ndarray[
                                                tuple[int], dtype[Any]] | Any, floating[Any]]:
    """
    Compute the average raw acceleration vector and its magnitude using trimmed mean.
    """
    if not 0.0 <= trim_percentage <= 0.5:
        raise ValueError("trim_percentage must be between 0.0 and 0.5")

    num_measurements = len(measurements)
    if num_measurements == 0:
        return np.zeros(3, dtype=np.float64), 0.0

    vectors = np.array([m.vec for m in measurements])

    trimmed_vectors_list = [] # Changed from pre-allocated numpy array to list
    for i in range(3): # Iterate over x, y, z components
        component_values = vectors[:, i]
        sorted_indices = np.argsort(component_values)
        trim_amount = int(num_measurements * trim_percentage)
        trimmed_indices = sorted_indices[trim_amount:num_measurements - trim_amount]
        trimmed_vectors_list.append(vectors[trimmed_indices, i]) # Append trimmed component values

    # Stack the trimmed components to form trimmed_vectors
    if trimmed_vectors_list: # Check if trimmed_vectors_list is not empty
        trimmed_vectors = np.column_stack(trimmed_vectors_list) # Stack columns
    else:
        return np.zeros(3, dtype=np.float64), 0.0 # Return zero vector and magnitude if no data after trimming


    sum_vec = np.sum(trimmed_vectors, axis=0)
    iterations = trimmed_vectors.shape[0] # Number of rows in trimmed_vectors is the count of trimmed measurements

    if iterations > 0:
        avg_vec = sum_vec / iterations
    else:
        avg_vec = np.zeros(3, dtype=np.float64)

    gravity_mag = np.linalg.norm(avg_vec)
    if gravity_mag != 0:
        g_normalized = avg_vec / gravity_mag
    else:
        g_normalized = avg_vec
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
    R_calib = Rotation.from_matrix(cls_1=None, matrix=R_matrix)
    return R_calib, gravity_mag # Return the Rotation object
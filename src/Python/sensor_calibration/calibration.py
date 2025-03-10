import numpy as np


class VectorMeasurement:
    def __init__(self, a: float, b: float, c: float):
        self.vec = np.array([a, b, c], dtype=np.float64)

    def normalize(self):
        magnitude = np.linalg.norm(self.vec)
        if magnitude != 0:
            self.vec /= magnitude



def calculate_avg_accel(measurements, iterations):
    sum_vec = np.zeros(3, dtype=np.float64)

    for measurement in measurements:
        measurement.normalize()
        sum_vec += measurement.vec

    avg_vec = sum_vec / iterations
    return avg_vec / np.linalg.norm(avg_vec)  # Ensure the result is a unit vector


def compute_rotation_matrix(g_normalized):
    v = np.array([g_normalized[1], -g_normalized[0], 0], dtype=np.float64)
    theta = np.arccos(g_normalized[2])

    v_norm = np.linalg.norm(v)
    if v_norm > 1e-6:
        v /= v_norm
    else:
        return np.identity(3)

    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])

    R = np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def calibrate_sensors(measurements, iterations):
    g_normalized = calculate_avg_accel(measurements, iterations)
    r = compute_rotation_matrix(g_normalized)

    return r
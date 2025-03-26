import numpy as np


def compare_rotations(R1, R2, tol=1e-2):
    """
    Compare two rotation matrices.
    Prints if they are nearly identical or not.
    """
    diff = np.linalg.norm(R1 - R2)
    if diff < tol:
        print("Rotation matrices are nearly identical.")
    else:
        print(f"Rotation matrices differ, difference norm: {diff:.4f}")

def rotation_difference_angle(R1, R2):
    """
    Computes the angular difference (in degrees) between two rotation matrices.
    """
    R_diff = R1.T @ R2
    angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    return np.degrees(angle_rad)

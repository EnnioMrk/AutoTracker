import numpy as np

from src.py.configs.config import VELOCITY_INTERVAL, VELOCITY_BUFFER


def analyze_velocity(acceleration_values):
    # Calculate velocity from acceleration using trapezoidal integration
    # acceleration_values should be in m/s²
    # sampling_time is the time between samples in seconds

    # Filter out any extreme outliers if necessary
    # acceleration_values = np.clip(acceleration_values, negative threshold, threshold)

    # Calculate velocity through integration (trapezoidal rule)
    sampling_time = VELOCITY_INTERVAL / len(acceleration_values)
    velocity = np.trapezoid(acceleration_values, dx=sampling_time)

    # Convert to km/h
    velocity_kmh = velocity * 3.6

    return velocity_kmh

def compare_velocity(v_0, acceleration_values):
    velocity = v_0 + analyze_velocity(acceleration_values)

    if velocity > v_0 + VELOCITY_BUFFER:
        print("Accelerating")
    elif velocity < v_0 - VELOCITY_BUFFER:
        print("Decelerating")

    return velocity
import matplotlib.pyplot as plt
import numpy as np


class DataVisualizer:
    def __init__(self, max_iterations=20000):
        """
        Initialize the visualizer to store sensor data and plot after max_iterations.

        Parameters:
            max_iterations (int): The number of iterations to collect data before plotting.
        """
        self.max_iterations = max_iterations
        self.iteration = 0

        # Lists to store the data over time.
        self.timestamps = []  # Time or iteration index
        self.baseline_rotated_acc = []  # Accelerometer data rotated by the baseline rotation (R_gravity)
        self.drifted_rotated_acc = []  # Accelerometer data rotated by the drifted/current orientation
        self.raw_acc = []  # Raw accelerometer data (optional)
        self.raw_gyro = []  # Raw gyroscope data (optional)
        self.dt_data = []  # Delta time values for each iteration

    def add_data(self, baseline_rotated, drifted_rotated, raw_acc, raw_gyro, dt, timestamp=None):
        """
        Add a new set of sensor data.

        Parameters:
            timestamp (float): The time stamp or iteration index.
            baseline_rotated (np.ndarray): Acceleration rotated by the baseline rotation.
            drifted_rotated (np.ndarray): Acceleration rotated by the updated orientation (including drift).
            raw_acc (np.ndarray): Raw accelerometer values.
            raw_gyro (np.ndarray): Raw gyroscope values.
            dt (float): The elapsed time since the previous data point.
        """
        if timestamp is None:
            timestamp = self.iteration  # Use iteration index as timestamp if not provided
        self.timestamps.append(timestamp)
        self.baseline_rotated_acc.append(baseline_rotated)
        self.drifted_rotated_acc.append(drifted_rotated)
        self.raw_acc.append(raw_acc)
        self.raw_gyro.append(raw_gyro)
        self.dt_data.append(dt)
        self.iteration += 1

        if self.max_iterations  == 0:
            print(f"Done {self.iteration} iterations")

        if self.iteration >= self.max_iterations:
            return False
        elif self.iteration % (self.max_iterations // 10) == 0:  # Every 10%
            print(f"Progress: {self.iteration / self.max_iterations * 100:.0f}% done")

        return True

    def create_graphs(self):
        """
        Generate plots comparing the baseline rotated acceleration and the drifted rotated acceleration.
        """
        # Convert lists to numpy arrays for easier indexing and plotting.
        times = np.array(self.timestamps)
        baseline_acc = np.array(self.baseline_rotated_acc)
        drifted_acc = np.array(self.drifted_rotated_acc)

        # Create subplots for each acceleration component (X, Y, Z).
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        components = ['X', 'Y', 'Z']
        for i in range(3):
            axes[i].plot(times, baseline_acc[:, i], label='Baseline Rotated Acc')
            axes[i].plot(times, drifted_acc[:, i], label='Drifted Rotated Acc', linestyle='--')
            axes[i].set_ylabel(f'{components[i]} Acc (m/s²)')
            axes[i].legend()
        axes[-1].set_xlabel('Time (s) or Iteration')
        plt.suptitle('Comparison of Baseline vs. Drifted Rotated Accelerations')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def clear_data(self):
        """
        Optionally clear the stored data after plotting.
        """
        self.timestamps = []
        self.baseline_rotated_acc = []
        self.drifted_rotated_acc = []
        self.raw_acc = []
        self.raw_gyro = []
        self.dt_data = []
        self.iteration = 0

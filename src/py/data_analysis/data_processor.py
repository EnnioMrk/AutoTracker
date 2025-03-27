from typing import List

import numpy as np

from scipy.stats import skew
from scipy.stats import kurtosis
from src.py.configs.filter_config import FilterConfig
from src.py.filter.complementary_filter import ComplementaryFilter
from src.py.filter.low_pass_filter import LowPassFilter
from src.py.record_data.file_manager import FileManager


class DataProcessor:
    def __init__(
            self,
            file_manager: FileManager,
            filter_config:FilterConfig,
            gyro_bias=None,
            accel_bias=None
    ):
        self.file_manager = file_manager
        self.config = filter_config
        self.gyro_bias = gyro_bias
        self.accel_bias = accel_bias
        self.calibrated = False
        self.low_pass_filter_acc = LowPassFilter(filter_config.alpha_low_pass)
        self.low_pass_filter_gyro = LowPassFilter(filter_config.alpha_low_pass)
        self.complementary_filter = ComplementaryFilter(filter_config.alpha_complementary)

        self.R_gravity = None
        self.gravity_mag = None

        self.position = np.zeros(3)
        self.velocity = 0

    def process(self, gyro, acc, dt):
        corrected_gyro = gyro - self.gyro_bias
        corrected_acc = acc - self.accel_bias
        filtered_gyro = self.low_pass_filter_gyro.filter(corrected_gyro)
        filtered_acc = self.low_pass_filter_acc.filter(corrected_acc)
        rotated_acc = self.R_gravity.apply(filtered_acc)

        if self.file_manager.recording:
            self.file_manager.update({
                "raw": [*acc, *gyro, dt]
            })

        return filtered_gyro, rotated_acc

    def update_bias(self, gyro_bias=None, accel_bias=None):
        if gyro_bias is None and accel_bias is None:
            print("No bias values provided.")
            return
        self.gyro_bias = gyro_bias
        self.accel_bias = accel_bias

    def is_stationary(self, accel_values, gyro_values, accel_stationary_low=0.95, accel_stationary_high=1.05,
                      gyro_stationary_threshold=2):
        """
        Checks if the sensor is stationary based on accelerometer and gyroscope readings.

        Args:
            accel_values (np.array): [x_accel, y_accel, z_accel]
            gyro_values (np.array): [x_gyro, y_gyro, z_gyro]
            accel_stationary_low (float): Lower bound for stationary acceleration magnitude.
            accel_stationary_high (float): Upper bound for stationary acceleration magnitude.
            gyro_stationary_threshold (float): Maximum magnitude for stationary gyroscope readings.

        Returns:
            bool: True if stationary, False otherwise.
        """
        corrected_gyro = gyro_values - self.gyro_bias
        corrected_acc = accel_values - self.accel_bias

        filtered_gyro = self.low_pass_filter_gyro.filter(corrected_gyro)
        filtered_acc = self.low_pass_filter_acc.filter(corrected_acc)

        accel_magnitude = np.linalg.norm(filtered_acc)
        gyro_magnitude = np.linalg.norm(filtered_gyro)

        is_accel_stationary = accel_stationary_low < accel_magnitude < accel_stationary_high
        is_gyro_stationary = gyro_magnitude < gyro_stationary_threshold

        if not is_accel_stationary:
            print(f"Accelerometer not stationary: {accel_magnitude}")
        else:
            self.velocity = 0
        if not is_gyro_stationary:
            print(f"Gyroscope not stationary: {gyro_magnitude}")

        return is_accel_stationary and is_gyro_stationary

    def process_data(self):
        pass

    def create_intervals(self):
        time_intervals = []
        constant_intervals = []
        all_values = []
        zero_crossing_intervals = []

    def analyze_intervals(self, intervals: List[np.array], zrc=True):
        for interval in intervals:
            ax, ay, az, gx, gy, gz, dt = [], [], [], [], [], [], []
            for values in interval:
                ax.append(values[0])
                ay.append(values[1])
                az.append(values[2])
                gx.append(values[3])
                gy.append(values[4])
                gz.append(values[5])
                dt.append(values[6])  # Assume dt is in each row of the interval

            # Process data
            gyro = np.array([gx, gy, gz])
            acc = np.array([ax, ay, az])
            dt = np.array(dt)
            duration = np.sum(dt)

            # Process acceleration
            self.process_acceleration(acc, dt, zrc)

    @staticmethod
    def zero_crossing_rate(data: np.ndarray) -> dict:
        """
        Computes zero-crossing rates relative to both mean and absolute zero.

        Parameters:
            data (np.ndarray): 1D array of numeric data (e.g., preprocessed accel axis).

        Returns:
            dict: A dictionary containing zero-crossing rates with two methods:
                - 'mean_crossing': Zero crossings relative to the window's mean
                - 'zero_crossing': Zero crossings relative to absolute zero
                Returns 0.0 for both if data length < 2.
        """
        data = np.asarray(data, dtype=float)

        if len(data) < 2:
            return {
                'mean_crossing': 0.0,
                'zero_crossing': 0.0
            }

        # Calculate crossings relative to the mean
        reference_point_mean = np.mean(data)
        ref_data_mean = data - reference_point_mean
        signs_mean = np.where(ref_data_mean >= 0, 1, -1)
        sign_changes_mean = np.diff(signs_mean)
        num_zero_crossings_mean = np.sum(sign_changes_mean != 0)
        normalized_rate_mean = num_zero_crossings_mean / (len(data) - 1)

        # Calculate crossings relative to absolute zero
        signs_zero = np.where(data >= 0, 1, -1)
        sign_changes_zero = np.diff(signs_zero)
        num_zero_crossings_zero = np.sum(sign_changes_zero != 0)
        normalized_rate_zero = num_zero_crossings_zero / (len(data) - 1)

        return {
            'mean_crossing': normalized_rate_mean,
            'zero_crossing': normalized_rate_zero
        }

    def process_acceleration(self, acc, dt, zrc=True):
        """
        Processes preprocessed accelerometer data (and time intervals) to extract features.

        Assumes bias/offset removal and filtering have already been applied to 'acc'.

        Args:
            acc (numpy.ndarray): Preprocessed accelerometer data, expected shape (3, N),
                                 where N is the number of samples.
                                 Verify axis order (e.g., Z, Y, X or X, Y, Z) matches your input.
            dt (list or numpy.ndarray): Time intervals between consecutive samples.
                                        dt[i] is the time delta between acc[:, i]
                                        and acc[:, i-1]. Must have length N.
            zrc (bool): Whether to calculate the Zero Crossing Rate.

        Returns:
            dict: A dictionary containing extracted features, organized by
                  'overall', 'x', 'y', 'z', and 'smv'. Returns None if
                  input data is too short for derivative calculations.
        """
        if not isinstance(acc, np.ndarray):
            acc = np.array(acc)
        if not isinstance(dt, np.ndarray):
            dt = np.array(dt)

        if acc.shape[0] != 3:
            raise ValueError(f"Input 'acc' should have 3 rows (axes), but got shape {acc.shape}")

        num_samples = acc.shape[1]

        if len(dt) != num_samples:
            raise ValueError(f"Length of dt array ({len(dt)}) must match the number of "
                             f"acceleration samples ({num_samples}).")

        if num_samples < 2:
            print("Warning: Too few samples (< 2) to calculate derivatives/diffs. Returning None.")
            return None

        # !!! IMPORTANT: Verify axis order here matches your 'acc' input !!!
        # Example assuming Z, Y, X order:
        az, ay, ax = acc
        # If your order is X, Y, Z, use:
        # ax, ay, az = acc

        # Prepare dt_intervals for derivatives (length N-1)
        dt_intervals = dt[1:]
        # Handle potential zero or negative time intervals
        if np.any(dt_intervals <= 0):
            print("Warning: dt contains non-positive values after the first element. "
                  "Replacing with small epsilon (1e-9) for derivative calculations.")
            dt_intervals = np.copy(dt_intervals)  # Avoid modifying original dt
            dt_intervals[dt_intervals <= 0] = 1e-9

        features_x = {}
        features_y = {}
        features_z = {}

        axis_mapping = [(ax, features_x, 'x'), (ay, features_y, 'y'), (az, features_z, 'z')]

        for axis_data, features_dict, axis_name in axis_mapping:
            features_dict['mean'] = np.mean(axis_data)
            features_dict['median'] = np.median(axis_data)
            features_dict['std'] = np.std(axis_data)
            features_dict['variance'] = np.var(axis_data)
            features_dict['max'] = np.max(axis_data)
            features_dict['min'] = np.min(axis_data)
            features_dict['range'] = features_dict['max'] - features_dict['min']
            features_dict['iqr'] = np.percentile(axis_data, 75) - np.percentile(axis_data, 25)
            features_dict['rms'] = np.sqrt(np.mean(np.square(axis_data)))
            features_dict['skewness'] = skew(axis_data)
            features_dict['kurtosis'] = kurtosis(axis_data)
            features_dict['sma'] = np.sum(np.abs(axis_data))
            if zrc:
                zero_crossing_dict = self.zero_crossing_rate(axis_data)
                features_dict['mean_zero_crossings'] = zero_crossing_dict['mean_crossing']
                features_dict['absolute_zero_crossings'] = zero_crossing_dict['zero_crossing']
            else:
                features_dict['mean_zero_crossings'] = None
                features_dict['absolute_zero_crossings'] = None

        diff_ax = np.diff(ax)
        diff_ay = np.diff(ay)
        diff_az = np.diff(az)

        jerk_x = diff_ax / dt_intervals
        jerk_y = diff_ay / dt_intervals
        jerk_z = diff_az / dt_intervals

        jerk_magnitude = np.sqrt(jerk_x ** 2 + jerk_y ** 2 + jerk_z ** 2)

        jerk_stats = {
            'jerk_mean_mag': np.mean(jerk_magnitude),
            'jerk_median_mag': np.median(jerk_magnitude),
            'jerk_std_mag': np.std(jerk_magnitude),
            'jerk_variance_mag': np.var(jerk_magnitude),
            'jerk_max_mag': np.max(jerk_magnitude),
            'jerk_min_mag': np.min(jerk_magnitude),
            'jerk_range_mag': np.max(jerk_magnitude) - np.min(jerk_magnitude),
            'jerk_iqr_mag': np.percentile(jerk_magnitude, 75) - np.percentile(jerk_magnitude, 25),
            'jerk_rms_mag': np.sqrt(np.mean(np.square(jerk_magnitude))),
            'jerk_skewness_mag': skew(jerk_magnitude),
            'jerk_kurtosis_mag': kurtosis(jerk_magnitude)
        }

        smv = np.linalg.norm(acc, axis=0)

        smv_stats = {
            'smv_mean': np.mean(smv),
            'smv_median': np.median(smv),
            'smv_std': np.std(smv),
            'smv_variance': np.var(smv),
            'smv_max': np.max(smv),
            'smv_min': np.min(smv),
            'smv_range': np.max(smv) - np.min(smv),
            'smv_iqr': np.percentile(smv, 75) - np.percentile(smv, 25),
            'smv_rms': np.sqrt(np.mean(np.square(smv))),
            'smv_skewness': skew(smv),
            'smv_kurtosis': kurtosis(smv),
            'smv_sma': np.sum(smv)  # SMA can also be defined on SMV
        }

        smv_diff = np.diff(smv) / dt_intervals

        smv_diff_stats = {
            'smv_mean_diff': np.mean(smv_diff),
            'smv_median_diff': np.median(smv_diff),
            'smv_std_diff': np.std(smv_diff),
            'smv_variance_diff': np.var(smv_diff),
            'smv_max_diff': np.max(smv_diff),
            'smv_min_diff': np.min(smv_diff),
            'smv_range_diff': np.max(smv_diff) - np.min(smv_diff),
            'smv_iqr_diff': np.percentile(smv_diff, 75) - np.percentile(smv_diff, 25),
            'smv_rms_diff': np.sqrt(np.mean(np.square(smv_diff))),
            'smv_skewness_diff': skew(smv_diff),
            'smv_kurtosis_diff': kurtosis(smv_diff)
        }

        corr_xy = np.nan_to_num(np.corrcoef(ax, ay)[0, 1])
        corr_xz = np.nan_to_num(np.corrcoef(ax, az)[0, 1])
        corr_yz = np.nan_to_num(np.corrcoef(ay, az)[0, 1])

        correlation_stats = {
            'corr_xy': corr_xy,
            'corr_xz': corr_xz,
            'corr_yz': corr_yz
        }

        overall_sma_axes = np.sum(np.abs(acc))

        final_features = {
            'overall': {
                'sma_axes_total': overall_sma_axes,
                **jerk_stats,
                **correlation_stats,
                **smv_stats,
                **smv_diff_stats
            },
            'x': features_x,
            'y': features_y,
            'z': features_z,
        }

        flattened_features = {}
        for category, values in final_features.items():
            if category == 'overall':
                flattened_features.update(values)
            else:
                for feature_name, value in values.items():
                    flattened_features[f"{category}_{feature_name}"] = value

        return final_features, flattened_features
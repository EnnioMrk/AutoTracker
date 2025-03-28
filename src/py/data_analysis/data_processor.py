from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import skew, kurtosis

from src.py.configs.data_collector import MainConfig
from src.py.configs.filter_config import FilterConfig
from src.py.data_analysis.visualizer import DataVisualizer
from src.py.filter.low_pass_filter import LowPassFilter
from src.py.record_data.file_manager import FileManager


class DataProcessor:
    """
    Handles processing of IMU sensor data including calibration application,
    filtering, rotation estimation, and feature extraction preparation.
    """
    def __init__(
            self,
            file_manager: FileManager,
            filter_config: FilterConfig,
            config: MainConfig = None,
            visualizer: DataVisualizer = None,
    ):
        """
        Initializes the DataProcessor.

        Args:
            file_manager: Instance of FileManager for recording data.
            filter_config: Configuration object for filter parameters.
            config: Main configuration object.
            visualizer: Optional DataVisualizer instance for debugging.
        """
        self.file_manager = file_manager
        self.config_filters = filter_config
        self.main_config = config or MainConfig()
        self.visualizer = visualizer

        # Calibration parameters - initialized to None, set via set_calibration
        self.gyro_bias: Optional[np.ndarray] = None
        self.accel_bias: Optional[np.ndarray] = None
        self.R_gravity: Optional[Rotation] = None
        self.gravity_mag: Optional[float] = None

        self.calibrated = False # Flag indicating if biases and R_gravity are set

        # Filters (using parameters from filter_config)
        self.low_pass_filter_acc = LowPassFilter(self.config_filters.alpha_low_pass)
        self.low_pass_filter_gyro = LowPassFilter(self.config_filters.alpha_low_pass)

        # State variables updated during processing
        self.current_rotation: Optional[Rotation] = None # Dynamic, gyro-integrated rotation
        self.position = np.zeros(3)
        self.velocity = np.zeros(3) # Velocity as a 3D vector

        self.time_interval_len: float = 2.0
        self.constant_interval_len: int = 200

    def process(self, gyro: List[float], acc: List[float], dt: float) -> bool | None | tuple[Any, Any]:
        """
        Processes a single raw sensor reading during live data collection.
        Applies bias, filters, updates rotation, and optionally records.

        Args:
            gyro: Raw gyroscope data [gx, gy, gz] (expected units: deg/s or rad/s).
            acc: Raw accelerometer data [ax, ay, az] (expected units: m/s² or g).
            dt: Time delta since last reading (seconds).

        Returns:
            Tuple of (filtered_gyro, rotated_acc_gravity) if successful,
            None if calibration is not done.
        """
        if not self.calibrated or self.accel_bias is None or self.gyro_bias is None or self.R_gravity is None:
            # print("Warning: Process called before calibration data is set. Skipping.")
            return None

        # Convert lists to numpy arrays
        acc_np = np.array(acc)
        gyro_np = np.array(gyro)
        gyro_np = np.radians(gyro_np)

        # 1. Apply Bias Correction
        corrected_gyro = gyro_np - self.gyro_bias
        corrected_acc = acc_np - self.accel_bias

        # 2. Apply Filters
        filtered_gyro = self.low_pass_filter_gyro.filter(corrected_gyro)
        filtered_acc = self.low_pass_filter_acc.filter(corrected_acc)

        # 3. Apply static gravity rotation to filtered acceleration
        # This gives acceleration in a world-aligned frame (Z-up), based on initial calibration
        rotated_acc_gravity = self.R_gravity.apply(filtered_acc)

        # 4. Update dynamic rotation and basic dead reckoning state
        self.update(filtered_gyro, filtered_acc, dt)

        # 5. Calculate acceleration rotated by the *current* dynamic orientation (for comparison/debug)
        updated_rotated_acc = np.array([np.nan]*3) # Initialize with NaN
        if self.current_rotation is not None:
             # This combines the initial gravity alignment with the accumulated gyro drift
             current_orientation: Rotation = self.R_gravity * self.current_rotation
             updated_rotated_acc = current_orientation.apply(filtered_acc)

        # 6. Visualization (Optional)
        if self.main_config.visualize and self.visualizer:
            if not self.visualizer.add_data(
                baseline_rotated=rotated_acc_gravity, # Rotated only by static R_gravity
                drifted_rotated=updated_rotated_acc, # Rotated by R_gravity * current_rotation
                raw_acc=acc_np,
                raw_gyro=gyro_np, # Pass original raw gyro
                dt=dt
            ):
                print("Visualizer limit reached (may signal stop).")
                return True

        # 7. Recording (Optional)
        if self.file_manager.recording:
            # Get current dynamic rotation as quaternion [x, y, z, w]
            # Use identity quaternion if rotation hasn't been calculated yet
            current_quat = self.current_rotation.as_quat() if self.current_rotation is not None else [0.0, 0.0, 0.0, 1.0]

            # Prepare data row: 6 sensors + dt + 4 quaternion values = 11 elements
            # Use original raw sensor values for saving
            raw_data_to_save = [*acc, *gyro, dt, *current_quat]

            self.file_manager.update({"raw": raw_data_to_save})

        return filtered_gyro, rotated_acc_gravity # Return potentially useful processed values

    def update(self, filtered_gyro_rad: np.ndarray, filtered_acc: np.ndarray, dt: float):
        """
        Update the dynamic rotation estimate and basic dead reckoning (velocity, position).
        Uses FILTERED sensor data.

        Args:
            filtered_gyro_rad: Filtered, bias-corrected gyroscope data in **radians/s**.
            filtered_acc: Filtered, bias-corrected accelerometer data (e.g., m/s²).
            dt: Time delta (seconds).
        """
        # --- Update Dynamic Rotation ---
        rotation_vector = filtered_gyro_rad * dt
        try:
            delta_rotation = Rotation.from_rotvec(rotation_vector)
            if self.current_rotation is None:
                self.current_rotation = delta_rotation
            else:
                self.current_rotation = self.current_rotation * delta_rotation
        except Exception:
             # print(f"Warning: Could not create delta_rotation: {e}") # Can be noisy
             pass # Keep previous rotation if update fails

        # --- Basic Dead Reckoning (Known to be inaccurate) ---
        is_stat = self.is_stationary(filtered_acc, filtered_gyro_rad)

        if not is_stat:
             # Rotate filtered acceleration into the world frame using static R_gravity
             accel_world_frame = self.R_gravity.apply(filtered_acc)

             # Subtract estimated gravity vector in world frame
             gravity_vector_world = np.array([0, 0, self.gravity_mag if self.gravity_mag else 9.81])
             linear_accel_world = accel_world_frame - gravity_vector_world

             # Integrate linear acceleration to get velocity (world frame)
             self.velocity += linear_accel_world * dt
             # Integrate velocity to get position (world frame)
             self.position += self.velocity * dt
        else:
            # Reset velocity when stationary to mitigate drift during stops
            self.velocity = np.zeros(3)

    def update_bias(self, gyro_bias: Optional[np.ndarray] = None, accel_bias: Optional[np.ndarray] = None):
        """Updates the stored bias values and resets filters and dynamic state."""
        if gyro_bias is None and accel_bias is None:
            print("Update_bias called with no new bias values.")
            return

        print("Updating bias and resetting filters...")
        changed = False
        if gyro_bias is not None:
            self.gyro_bias = np.array(gyro_bias)
            self.low_pass_filter_gyro.reset()
            changed = True
            print(f"Gyro bias updated: {self.gyro_bias}")
        if accel_bias is not None:
            self.accel_bias = np.array(accel_bias)
            self.low_pass_filter_acc.reset()
            changed = True
            print(f"Accel bias updated: {self.accel_bias}")

        if changed:
             print("Resetting current rotation estimate and dead reckoning state due to bias update.")
             self.current_rotation = None
             self.velocity = np.zeros(3)
             self.position = np.zeros(3)

    def set_calibration(self, accel_bias: np.ndarray, gyro_bias: np.ndarray, R_gravity: Rotation, gravity_mag: float):
        """Sets all calibration parameters at once and resets state."""
        print("Setting full calibration parameters...")
        self.accel_bias = np.array(accel_bias)
        self.gyro_bias = np.array(gyro_bias)
        self.R_gravity = R_gravity
        self.gravity_mag = gravity_mag
        self.calibrated = True
        # Reset filters and dynamic state
        self.low_pass_filter_acc.reset()
        self.low_pass_filter_gyro.reset()
        self.current_rotation = None
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        print("Calibration set. Filters and dynamic state reset.")

    def is_stationary(self, current_accel: np.ndarray, current_gyro_rad: np.ndarray,
                       accel_magnitude_threshold: float = 0.2, # Allow slightly larger deviation (e.g., 0.2 m/s^2)
                       gyro_magnitude_threshold_dps: float = 2.0 # Threshold on gyro magnitude in deg/s
                      ) -> bool:
        """
        Checks if the sensor is likely stationary. Uses calibrated gravity magnitude.

        Args:
            current_accel: Current accelerometer reading (raw, corrected, or filtered) in m/s².
            current_gyro_rad: Current gyroscope reading (raw, corrected, or filtered) in **rad/s**.
            accel_magnitude_threshold: Max deviation of accel magnitude from calibrated gravity (m/s²).
            gyro_magnitude_threshold_dps: Max magnitude of gyro readings in **degrees/s**.

        Returns:
            bool: True if likely stationary, False otherwise.
        """
        if not self.calibrated or self.gravity_mag is None:
             return False # Cannot check reliably without calibration

        accel_mag = np.linalg.norm(current_accel)
        # Convert gyro magnitude to degrees per second for threshold comparison
        gyro_mag_dps = np.linalg.norm(np.degrees(current_gyro_rad))

        is_accel_stationary = abs(accel_mag - self.gravity_mag) < accel_magnitude_threshold
        is_gyro_stationary = gyro_mag_dps < gyro_magnitude_threshold_dps

        return is_accel_stationary and is_gyro_stationary

    # --- Static Methods for Offline Processing ---

    @staticmethod
    def reprocess_interval(interval_df: pd.DataFrame, calibration_data: Dict, filter_alpha: float = 0.2) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Reprocesses an interval of raw data using loaded calibration info.
        Applies bias correction and *stateful* filtering sequentially.
        Calculates derived data types (corrected, filtered, rotated).

        Args:
            interval_df (pd.DataFrame): DataFrame of raw data for one interval.
                                        Expected columns: ['acc_x', ..., 'dt', 'quat_x', ...].
            calibration_data (Dict): Dictionary loaded from JSON, containing
                                     'accel_bias', 'gyro_bias', 'R_gravity_quat'.
            filter_alpha (float): Alpha value for the LowPassFilters during reprocessing.

        Returns:
            Optional[Dict[str, pd.DataFrame]]: Dictionary mapping data types to DataFrames
                                               or None if reprocessing fails.
        """
        required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'dt',
                         'quat_x', 'quat_y', 'quat_z', 'quat_w']
        if not all(col in interval_df.columns for col in required_cols):
            print(f"Error: Interval DataFrame missing required columns for reprocessing. Need: {required_cols}")
            return None
        if interval_df.empty: return None

        try:
            accel_bias = np.array(calibration_data['accel_bias'])
            gyro_bias = np.array(calibration_data['gyro_bias'])
            R_gravity = Rotation.from_quat(calibration_data['R_gravity_quat'])
        except (KeyError, TypeError, ValueError) as e:
            print(f"Error loading/parsing calibration_data: {e}")
            return None

        lpf_acc = LowPassFilter(alpha=filter_alpha)
        lpf_gyro = LowPassFilter(alpha=filter_alpha)

        num_rows = len(interval_df)
        # Preallocate numpy arrays for efficiency
        corrected_acc_data = np.zeros((num_rows, 3))
        corrected_gyro_data = np.zeros((num_rows, 3))
        filtered_acc_data = np.zeros((num_rows, 3))
        filtered_gyro_data = np.zeros((num_rows, 3))
        rotated_gravity_acc_data = np.zeros((num_rows, 3))
        rotated_current_acc_data = np.zeros((num_rows, 3))

        raw_acc_np = interval_df[['acc_x', 'acc_y', 'acc_z']].values
        raw_gyro_np = interval_df[['gyro_x', 'gyro_y', 'gyro_z']].values
        quat_np = interval_df[['quat_x', 'quat_y', 'quat_z', 'quat_w']].values

        for i in range(num_rows):
            raw_acc = raw_acc_np[i]
            raw_gyro = raw_gyro_np[i]
            # --- TODO: Assume raw_gyro from file is in same units as needed (e.g. rad/s) ---
            # If raw_gyro is deg/s, convert it here before bias subtraction/filtering
            # raw_gyro = np.radians(raw_gyro)

            saved_quat = quat_np[i]
            try:
                 # Combine static gravity rotation with saved dynamic rotation
                 R_combined_saved = R_gravity * Rotation.from_quat(saved_quat)
            except ValueError:
                 R_combined_saved = Rotation.identity() # Fallback

            corrected_acc = raw_acc - accel_bias
            corrected_gyro = raw_gyro - gyro_bias

            filtered_acc = lpf_acc.filter(corrected_acc)
            filtered_gyro = lpf_gyro.filter(corrected_gyro)

            rotated_acc_gravity = R_gravity.apply(filtered_acc)
            rotated_acc_current = R_combined_saved.apply(filtered_acc) # Use combined rotation

            # Store results
            corrected_acc_data[i, :] = corrected_acc
            corrected_gyro_data[i, :] = corrected_gyro
            filtered_acc_data[i, :] = filtered_acc
            filtered_gyro_data[i, :] = filtered_gyro
            rotated_gravity_acc_data[i, :] = rotated_acc_gravity
            rotated_current_acc_data[i, :] = rotated_acc_current

        # Create DataFrames
        cols_acc = ['acc_x', 'acc_y', 'acc_z']
        cols_gyro = ['gyro_x', 'gyro_y', 'gyro_z']
        processed_data = {
            'raw_acc': pd.DataFrame(raw_acc_np, columns=cols_acc, index=interval_df.index),
            'raw_gyro': pd.DataFrame(raw_gyro_np, columns=cols_gyro, index=interval_df.index),
            'corrected_acc': pd.DataFrame(corrected_acc_data, columns=cols_acc, index=interval_df.index),
            'corrected_gyro': pd.DataFrame(corrected_gyro_data, columns=cols_gyro, index=interval_df.index),
            'filtered_acc': pd.DataFrame(filtered_acc_data, columns=cols_acc, index=interval_df.index),
            'filtered_gyro': pd.DataFrame(filtered_gyro_data, columns=cols_gyro, index=interval_df.index),
            'rotated_gravity': pd.DataFrame(rotated_gravity_acc_data, columns=cols_acc, index=interval_df.index),
            'rotated_current': pd.DataFrame(rotated_current_acc_data, columns=cols_acc, index=interval_df.index),
            'dt': interval_df['dt'].copy()
        }
        return processed_data

    @staticmethod
    def create_intervals(data: List[List[Any]]) -> Optional[Dict[str, List[List[List[Any]]]]]:
        """
        Creates intervals from sensor data based on time, sample count, and az zero-crossing events.
        Expects input data as list of lists [[ax,ay,az,gx,gy,gz,dt,...], ...].
        The interval data returned is still in the raw list-of-lists format.

        Args:
            data: List of lists containing raw sensor data rows (at least 7 elements needed).

        Returns:
            Dict mapping interval type ('time', 'sample', 'event') to lists of intervals,
            where each interval is a list of original data rows. Returns None if input is invalid.
        """
        if not data or not isinstance(data, list) or not isinstance(data[0], list):
            print("Warning: Input data is empty or not a list of lists. Cannot create intervals.")
            return None

        time_intervals = []
        sample_intervals = []
        event_intervals = []

        current_time_interval = []
        current_sample_interval = []
        current_event_interval = []

        current_time_duration = 0.0
        current_sample_count = 0
        prev_az = None

        # Default interval lengths (could be moved to config)
        TIME_INTERVAL_LEN = 2.0
        SAMPLE_INTERVAL_LEN = 200

        AZ_INDEX = 2 # Index for Z-acceleration in the inner list
        DT_INDEX = 6 # Index for dt in the inner list

        for i, row in enumerate(data):
             # Basic validation for each row
             if len(row) < max(AZ_INDEX, DT_INDEX) + 1:
                  print(f"Warning: Row {i} is too short. Skipping.")
                  continue
             try:
                  az = float(row[AZ_INDEX])
                  dt = float(row[DT_INDEX])
                  if dt < 0:
                       print(f"Warning: Negative dt ({dt}) found in row {i}. Skipping row.")
                       continue
             except (ValueError, TypeError) as e:
                  print(f"Warning: Invalid data types in row {i}. Skipping row (az={row[AZ_INDEX]}, dt={row[DT_INDEX]}). Error: {e}")
                  continue

             # Time-based interval
             current_time_interval.append(row)
             current_time_duration += dt
             if current_time_duration >= TIME_INTERVAL_LEN and len(current_time_interval) > 1:
                  time_intervals.append(list(current_time_interval))
                  current_time_interval = []
                  current_time_duration = 0.0

             # Sample-based interval
             current_sample_interval.append(row)
             current_sample_count += 1
             if current_sample_count == SAMPLE_INTERVAL_LEN:
                  sample_intervals.append(list(current_sample_interval))
                  current_sample_interval = []
                  current_sample_count = 0

             # Event-based interval (Z-axis zero-crossing)
             # Needs previous value to detect crossing
             is_crossing = False
             if prev_az is not None:
                  if (az > 0 and prev_az <= 0) or (az < 0 and prev_az >= 0):
                       is_crossing = True

             if is_crossing and current_event_interval: # Start new interval on crossing
                  # Only add if the previous interval wasn't empty
                  if len(current_event_interval) > 1:
                       event_intervals.append(list(current_event_interval))
                  current_event_interval = [row] # Start new interval with current row
             else:
                  current_event_interval.append(row)
             prev_az = az # Update previous value for next iteration

        # Add any remaining data in the current intervals
        if current_time_interval and len(current_time_interval) > 1: time_intervals.append(current_time_interval)
        if current_sample_interval and len(current_sample_interval) > 1: sample_intervals.append(current_sample_interval)
        if current_event_interval and len(current_event_interval) > 1: event_intervals.append(current_event_interval)

        return {'time': time_intervals, 'sample': sample_intervals, 'event': event_intervals}


    @staticmethod
    def zero_crossing_rate(data: np.ndarray) -> Dict[str, float]:
        """ Computes zero-crossing rates relative to mean and absolute zero. """
        data = np.asarray(data, dtype=float)
        # Handle potential NaNs if input data might contain them
        valid_data = data[np.isfinite(data)]
        if len(valid_data) < 2: return {'mean_crossing': 0.0, 'zero_crossing': 0.0}

        # Mean crossing
        reference_point_mean = np.mean(valid_data) # Use mean of valid data
        ref_data_mean = valid_data - reference_point_mean
        signs_mean = np.where(ref_data_mean >= 0, 1, -1)
        sign_changes_mean = np.diff(signs_mean)
        num_zero_crossings_mean = np.sum(sign_changes_mean != 0)
        normalized_rate_mean = num_zero_crossings_mean / (len(valid_data) - 1)

        # Absolute zero crossing
        signs_zero = np.where(valid_data >= 0, 1, -1)
        sign_changes_zero = np.diff(signs_zero)
        num_zero_crossings_zero = np.sum(sign_changes_zero != 0)
        normalized_rate_zero = num_zero_crossings_zero / (len(valid_data) - 1)

        return {'mean_crossing': normalized_rate_mean, 'zero_crossing': normalized_rate_zero}


    @staticmethod
    def process_acceleration(acc: np.ndarray, dt: np.ndarray, zrc: bool = True) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """ Processes accelerometer data (3, N) and dt (N,) to extract features. """
        # (Input validation as before)
        if not isinstance(acc, np.ndarray) or acc.shape[0] != 3 or not isinstance(dt, np.ndarray):
            raise ValueError("Invalid input types or shapes for process_acceleration")
        num_samples = acc.shape[1]
        if len(dt) != num_samples: raise ValueError("Length mismatch between acc and dt")
        if num_samples < 2: return None

        ax, ay, az = acc
        dt_intervals = dt[1:]
        valid_dt_mask = dt_intervals > 1e-9
        if not np.all(valid_dt_mask):
            dt_intervals = np.copy(dt_intervals)
            dt_intervals[~valid_dt_mask] = 1e-9 # Replace non-positive dt with small value

        features_x, features_y, features_z = {}, {}, {}
        axis_mapping = [(ax, features_x, 'x'), (ay, features_y, 'y'), (az, features_z, 'z')]

        # Calculate axis-wise features robustly
        for axis_data, features_dict, axis_name in axis_mapping:
            if axis_data.size == 0: continue
            valid_axis_data = axis_data[np.isfinite(axis_data)] # Use only finite values
            if valid_axis_data.size == 0: continue # Skip if no valid data

            axis_std = np.std(valid_axis_data)
            features_dict['mean'] = np.mean(valid_axis_data)
            features_dict['median'] = np.median(valid_axis_data)
            features_dict['std'] = axis_std
            features_dict['variance'] = np.var(valid_axis_data)
            features_dict['max'] = np.max(valid_axis_data)
            features_dict['min'] = np.min(valid_axis_data)
            features_dict['range'] = features_dict['max'] - features_dict['min']
            if valid_axis_data.size >= 4:
                 q75, q25 = np.percentile(valid_axis_data, [75, 25])
                 features_dict['iqr'] = q75 - q25
            else: features_dict['iqr'] = 0.0 # Or NaN
            features_dict['rms'] = np.sqrt(np.mean(np.square(valid_axis_data)))
            features_dict['skewness'] = skew(valid_axis_data) if axis_std > 1e-9 and len(valid_axis_data)>2 else 0.0
            features_dict['kurtosis'] = kurtosis(valid_axis_data) if axis_std > 1e-9 and len(valid_axis_data)>3 else 0.0
            features_dict['sma'] = np.sum(np.abs(valid_axis_data))
            if zrc:
                zcr_res = DataProcessor.zero_crossing_rate(axis_data) # Pass original with NaNs if ZCR handles it, or valid_axis_data
                features_dict['mean_zero_crossings'] = zcr_res['mean_crossing']
                features_dict['absolute_zero_crossings'] = zcr_res['zero_crossing']

        # Calculate Jerk features robustly
        jerk_stats = {}
        if dt_intervals.size > 0:
             diff_ax = np.diff(ax)
             diff_ay = np.diff(ay)
             diff_az = np.diff(az)
             jerk_x = diff_ax / dt_intervals
             jerk_y = diff_ay / dt_intervals
             jerk_z = diff_az / dt_intervals
             jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
             valid_jerk = jerk_magnitude[np.isfinite(jerk_magnitude)]
             if valid_jerk.size > 0:
                  jerk_std = np.std(valid_jerk)
                  jerk_stats = { # Calculate stats on valid_jerk
                       'jerk_mean_mag': np.mean(valid_jerk), 'jerk_median_mag': np.median(valid_jerk),
                       'jerk_std_mag': jerk_std, 'jerk_variance_mag': np.var(valid_jerk),
                       'jerk_max_mag': np.max(valid_jerk), 'jerk_min_mag': np.min(valid_jerk),
                       'jerk_range_mag': np.ptp(valid_jerk), # peak-to-peak
                       'jerk_iqr_mag': np.percentile(valid_jerk, 75) - np.percentile(valid_jerk, 25) if valid_jerk.size>=4 else 0.0,
                       'jerk_rms_mag': np.sqrt(np.mean(np.square(valid_jerk))),
                       'jerk_skewness_mag': skew(valid_jerk) if jerk_std > 1e-9 and len(valid_jerk)>2 else 0.0,
                       'jerk_kurtosis_mag': kurtosis(valid_jerk) if jerk_std > 1e-9 and len(valid_jerk)>3 else 0.0
                  }

        # Calculate SMV features robustly
        smv_stats = {}
        smv = np.linalg.norm(acc, axis=0)
        valid_smv = smv[np.isfinite(smv)]
        if valid_smv.size > 0:
             smv_std = np.std(valid_smv)
             smv_stats = { # Calculate stats on valid_smv
                  'smv_mean': np.mean(valid_smv), 'smv_median': np.median(valid_smv), 'smv_std': smv_std,
                  'smv_variance': np.var(valid_smv), 'smv_max': np.max(valid_smv), 'smv_min': np.min(valid_smv),
                  'smv_range': np.ptp(valid_smv),
                  'smv_iqr': np.percentile(valid_smv, 75) - np.percentile(valid_smv, 25) if valid_smv.size>=4 else 0.0,
                  'smv_rms': np.sqrt(np.mean(np.square(valid_smv))),
                  'smv_skewness': skew(valid_smv) if smv_std > 1e-9 and len(valid_smv)>2 else 0.0,
                  'smv_kurtosis': kurtosis(valid_smv) if smv_std > 1e-9 and len(valid_smv)>3 else 0.0,
                  'smv_sma': np.sum(valid_smv) # Sum of valid magnitudes
             }

        # Calculate SMV Diff features robustly
        smv_diff_stats = {}
        if dt_intervals.size > 0 and smv.size > 1:
             smv_diff = np.diff(smv) / dt_intervals
             valid_smv_diff = smv_diff[np.isfinite(smv_diff)]
             if valid_smv_diff.size > 0:
                  smv_diff_std = np.std(valid_smv_diff)
                  smv_diff_stats = { # Calculate stats on valid_smv_diff
                       'smv_mean_diff': np.mean(valid_smv_diff), 'smv_median_diff': np.median(valid_smv_diff),
                       'smv_std_diff': smv_diff_std, 'smv_variance_diff': np.var(valid_smv_diff),
                       'smv_max_diff': np.max(valid_smv_diff), 'smv_min_diff': np.min(valid_smv_diff),
                       'smv_range_diff': np.ptp(valid_smv_diff),
                       'smv_iqr_diff': np.percentile(valid_smv_diff, 75) - np.percentile(valid_smv_diff, 25) if valid_smv_diff.size>=4 else 0.0,
                       'smv_rms_diff': np.sqrt(np.mean(np.square(valid_smv_diff))),
                       'smv_skewness_diff': skew(valid_smv_diff) if smv_diff_std > 1e-9 and len(valid_smv_diff)>2 else 0.0,
                       'smv_kurtosis_diff': kurtosis(valid_smv_diff) if smv_diff_std > 1e-9 and len(valid_smv_diff)>3 else 0.0
                  }

        # Calculate Correlation features robustly
        correlation_stats = {}
        valid_mask = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
        if np.sum(valid_mask) > 1: # Need more than 1 valid point for correlation
            ax_valid, ay_valid, az_valid = ax[valid_mask], ay[valid_mask], az[valid_mask]
            std_ax, std_ay, std_az = np.std(ax_valid), np.std(ay_valid), np.std(az_valid)
            corr_xy = np.corrcoef(ax_valid, ay_valid)[0, 1] if std_ax > 1e-9 and std_ay > 1e-9 else 0.0
            corr_xz = np.corrcoef(ax_valid, az_valid)[0, 1] if std_ax > 1e-9 and std_az > 1e-9 else 0.0
            corr_yz = np.corrcoef(ay_valid, az_valid)[0, 1] if std_ay > 1e-9 and std_az > 1e-9 else 0.0
            correlation_stats = {'corr_xy': corr_xy, 'corr_xz': corr_xz, 'corr_yz': corr_yz}

        # Assemble results
        overall_sma_axes = features_x.get('sma', 0) + features_y.get('sma', 0) + features_z.get('sma', 0)
        detailed_features = {
            'overall': {'sma_axes_total': overall_sma_axes, **jerk_stats, **correlation_stats, **smv_stats, **smv_diff_stats},
            'x': features_x, 'y': features_y, 'z': features_z,
        }
        flattened_features = {}
        for category, values in detailed_features.items():
            prefix = "acc_" + (category + "_" if category != 'overall' else "")
            for feature_name, value in values.items():
                # Ensure value is finite, replace with 0 if not
                flattened_features[f"{prefix}{feature_name}"] = float(value) if np.isfinite(value) else 0.0

        return detailed_features, flattened_features


    @staticmethod
    def process_gyro(gyro: np.ndarray, dt: np.ndarray, zrc: bool = True) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """ Processes gyroscope data (3, N) and dt (N,) to extract features. """
        # (Input validation similar to process_acceleration)
        if not isinstance(gyro, np.ndarray) or gyro.shape[0] != 3 or not isinstance(dt, np.ndarray):
            raise ValueError("Invalid input types or shapes for process_gyro")
        num_samples = gyro.shape[1]
        if len(dt) != num_samples: raise ValueError("Length mismatch between gyro and dt")
        if num_samples < 2: return None

        gx, gy, gz = gyro
        dt_intervals = dt[1:]
        valid_dt_mask = dt_intervals > 1e-9
        if not np.all(valid_dt_mask):
            dt_intervals = np.copy(dt_intervals)
            dt_intervals[~valid_dt_mask] = 1e-9 # Replace non-positive dt

        # --- Calculate axis-wise features (similar robust logic as in process_acceleration) ---
        features_x, features_y, features_z = {}, {}, {}
        # ... (loop through gx, gy, gz using nan-aware numpy functions) ...
        for axis_data, features_dict, axis_name in [(gx, features_x, 'x'), (gy, features_y, 'y'), (gz, features_z, 'z')]:
            if axis_data.size == 0: continue
            valid_axis_data = axis_data[np.isfinite(axis_data)]
            if valid_axis_data.size == 0: continue
            axis_std = np.std(valid_axis_data)
            features_dict['mean'] = np.mean(valid_axis_data)
            features_dict['median'] = np.median(valid_axis_data)
            features_dict['std'] = axis_std
            # ... (variance, max, min, range, iqr, rms, skew, kurt, sma, zcr) ...
            features_dict['variance']=np.var(valid_axis_data); features_dict['max']=np.max(valid_axis_data); features_dict['min']=np.min(valid_axis_data)
            features_dict['range']=np.ptp(valid_axis_data); features_dict['rms']=np.sqrt(np.mean(np.square(valid_axis_data)))
            features_dict['sma']=np.sum(np.abs(valid_axis_data))
            if valid_axis_data.size>=4: q75,q25=np.percentile(valid_axis_data,[75,25]); features_dict['iqr']=q75-q25
            else: features_dict['iqr']=0.0
            features_dict['skewness'] = skew(valid_axis_data) if axis_std > 1e-9 and len(valid_axis_data)>2 else 0.0
            features_dict['kurtosis'] = kurtosis(valid_axis_data) if axis_std > 1e-9 and len(valid_axis_data)>3 else 0.0
            if zrc: zcr_res = DataProcessor.zero_crossing_rate(axis_data); features_dict['mean_zero_crossings'] = zcr_res['mean_crossing']; features_dict['absolute_zero_crossings'] = zcr_res['zero_crossing']

        # --- Calculate Angular Acceleration features (robustly) ---
        ang_accel_stats = {}
        # ... (calculate diffs, divide by dt_intervals, calculate magnitude, use nan-aware stats) ...
        if dt_intervals.size > 0:
             diff_gx, diff_gy, diff_gz = np.diff(gx), np.diff(gy), np.diff(gz)
             ang_accel_x, ang_accel_y, ang_accel_z = diff_gx/dt_intervals, diff_gy/dt_intervals, diff_gz/dt_intervals
             ang_accel_mag = np.sqrt(ang_accel_x**2 + ang_accel_y**2 + ang_accel_z**2)
             valid_ang_accel = ang_accel_mag[np.isfinite(ang_accel_mag)]
             if valid_ang_accel.size > 0:
                  ang_accel_std = np.std(valid_ang_accel)
                  ang_accel_stats = { # Calculate stats on valid_ang_accel
                       'ang_accel_mean_mag': np.mean(valid_ang_accel), 'ang_accel_median_mag': np.median(valid_ang_accel),
                       'ang_accel_std_mag': ang_accel_std, 'ang_accel_variance_mag': np.var(valid_ang_accel),
                       'ang_accel_max_mag': np.max(valid_ang_accel), 'ang_accel_min_mag': np.min(valid_ang_accel),
                       'ang_accel_range_mag': np.ptp(valid_ang_accel),
                       'ang_accel_iqr_mag': np.percentile(valid_ang_accel, 75) - np.percentile(valid_ang_accel, 25) if valid_ang_accel.size>=4 else 0.0,
                       'ang_accel_rms_mag': np.sqrt(np.mean(np.square(valid_ang_accel))),
                       'ang_accel_skewness_mag': skew(valid_ang_accel) if ang_accel_std > 1e-9 and len(valid_ang_accel)>2 else 0.0,
                       'ang_accel_kurtosis_mag': kurtosis(valid_ang_accel) if ang_accel_std > 1e-9 and len(valid_ang_accel)>3 else 0.0
                  }

        # --- Calculate GMV (Gyro Magnitude Vector) features (robustly) ---
        gmv_stats = {}
        # ... (calculate norm, use nan-aware stats) ...
        gmv = np.linalg.norm(gyro, axis=0)
        valid_gmv = gmv[np.isfinite(gmv)]
        if valid_gmv.size > 0:
             gmv_std = np.std(valid_gmv)
             gmv_stats = { # Calculate stats on valid_gmv
                  'gmv_mean': np.mean(valid_gmv), 'gmv_median': np.median(valid_gmv), 'gmv_std': gmv_std,
                  'gmv_variance': np.var(valid_gmv), 'gmv_max': np.max(valid_gmv), 'gmv_min': np.min(valid_gmv),
                  'gmv_range': np.ptp(valid_gmv),
                  'gmv_iqr': np.percentile(valid_gmv, 75) - np.percentile(valid_gmv, 25) if valid_gmv.size>=4 else 0.0,
                  'gmv_rms': np.sqrt(np.mean(np.square(valid_gmv))),
                  'gmv_skewness': skew(valid_gmv) if gmv_std > 1e-9 and len(valid_gmv)>2 else 0.0,
                  'gmv_kurtosis': kurtosis(valid_gmv) if gmv_std > 1e-9 and len(valid_gmv)>3 else 0.0,
                  'gmv_sma': np.sum(valid_gmv)
             }


        # --- Calculate GMV Diff features (robustly) ---
        gmv_diff_stats = {}
        # ... (calculate diff, divide by dt_intervals, use nan-aware stats) ...
        if dt_intervals.size > 0 and gmv.size > 1:
             gmv_diff = np.diff(gmv) / dt_intervals
             valid_gmv_diff = gmv_diff[np.isfinite(gmv_diff)]
             if valid_gmv_diff.size > 0:
                  gmv_diff_std = np.std(valid_gmv_diff)
                  gmv_diff_stats = { # Calculate stats on valid_gmv_diff
                       'gmv_mean_diff': np.mean(valid_gmv_diff), 'gmv_median_diff': np.median(valid_gmv_diff),
                       'gmv_std_diff': gmv_diff_std, 'gmv_variance_diff': np.var(valid_gmv_diff),
                       'gmv_max_diff': np.max(valid_gmv_diff), 'gmv_min_diff': np.min(valid_gmv_diff),
                       'gmv_range_diff': np.ptp(valid_gmv_diff),
                       'gmv_iqr_diff': np.percentile(valid_gmv_diff, 75) - np.percentile(valid_gmv_diff, 25) if valid_gmv_diff.size>=4 else 0.0,
                       'gmv_rms_diff': np.sqrt(np.mean(np.square(valid_gmv_diff))),
                       'gmv_skewness_diff': skew(valid_gmv_diff) if gmv_diff_std > 1e-9 and len(valid_gmv_diff)>2 else 0.0,
                       'gmv_kurtosis_diff': kurtosis(valid_gmv_diff) if gmv_diff_std > 1e-9 and len(valid_gmv_diff)>3 else 0.0
                  }

        # --- Calculate Correlation features (robustly) ---
        correlation_stats = {}
        # ... (use valid_mask based on gx, gy, gz finiteness, calculate corrcoef) ...
        valid_mask = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(gz)
        if np.sum(valid_mask) > 1:
             gx_valid, gy_valid, gz_valid = gx[valid_mask], gy[valid_mask], gz[valid_mask]
             std_gx, std_gy, std_gz = np.std(gx_valid), np.std(gy_valid), np.std(gz_valid)
             corr_xy = np.corrcoef(gx_valid, gy_valid)[0, 1] if std_gx > 1e-9 and std_gy > 1e-9 else 0.0
             corr_xz = np.corrcoef(gx_valid, gz_valid)[0, 1] if std_gx > 1e-9 and std_gz > 1e-9 else 0.0
             corr_yz = np.corrcoef(gy_valid, gz_valid)[0, 1] if std_gy > 1e-9 and std_gz > 1e-9 else 0.0
             correlation_stats = {'corr_xy': corr_xy, 'corr_xz': corr_xz, 'corr_yz': corr_yz}

        # Assemble results
        overall_sma_axes = features_x.get('sma', 0) + features_y.get('sma', 0) + features_z.get('sma', 0)
        detailed_features = {
            'overall': {'sma_axes_total': overall_sma_axes, **ang_accel_stats, **correlation_stats, **gmv_stats, **gmv_diff_stats},
            'x': features_x, 'y': features_y, 'z': features_z,
        }
        flattened_features = {}
        for category, values in detailed_features.items():
            prefix = "gyro_" + (category + "_" if category != 'overall' else "")
            for feature_name, value in values.items():
                 # Ensure value is finite, replace with 0 if not
                flattened_features[f"{prefix}{feature_name}"] = float(value) if np.isfinite(value) else 0.0

        return detailed_features, flattened_features
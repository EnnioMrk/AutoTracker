# --- START OF FILE model.py ---

import json
import os
from typing import List, Any, Optional, Tuple, Dict
from collections import Counter # Needed for majority voting
import time # For timing folds

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # For confusion matrix plotting


try:
    # Assume running from project root or IDE handles paths
    from src.py.data_analysis.data_processor import DataProcessor
except ImportError:
    # Fallback if the structure is different during execution
    print("Warning: Could not import DataProcessor directly. Ensure paths are correct.")
    try:
        # Try relative import if run directly as script
        from data_processor import DataProcessor
    except ImportError:
        print("Error: Failed to import DataProcessor using fallback.")
        DataProcessor = None # Set to None to handle import error later


class StreetTextureClassifier:
    def __init__(self,
                 features: Optional[List[str]] = None,
                 model_type: str = 'random_forest',
                 zrc: bool = True,
                 feature_data_type: str = 'rotated_gravity',
                 random_state: int = 42):
        """
        Args:
            features (Optional[List[str]]): Specific features to use. If None, use all generated.
            model_type (str): Type of classifier model. Currently only 'random_forest'.
            zrc (bool): Whether to include Zero-Crossing Rate features.
            feature_data_type (str): Which processed data to use for feature extraction.
                                     Options: 'raw', 'corrected', 'filtered',
                                              'rotated_gravity', 'rotated_current'.
            random_state (int): Random seed for reproducibility.
        """
        self._user_features = features
        self._actual_features_used: Optional[List[str]] = None
        self.zrc = zrc
        self.random_state = random_state

        # Validate feature_data_type
        valid_feature_types = ['raw', 'corrected', 'filtered', 'rotated_gravity', 'rotated_current']
        if feature_data_type not in valid_feature_types:
            raise ValueError(f"Invalid feature_data_type: '{feature_data_type}'. "
                             f"Must be one of {valid_feature_types}")
        self.feature_data_type = feature_data_type
        # Suppress verbose init print
        # print(f"Classifier configured to extract features from: {self.feature_data_type} acceleration data.")

        self.scaler = StandardScaler()
        if model_type == 'random_forest':
            # Consider adding n_jobs=-1 for potentially faster training/prediction on multi-core CPUs
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced', n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.is_trained = False
        self.feature_importances_: Optional[pd.Series] = None
        self.classes_: Optional[np.ndarray] = None

    def _extract_features(self,
                           interval_df_raw: pd.DataFrame,
                           calibration_data: Dict
                           ) -> Optional[Dict[str, Any]]:
        """
        Reprocesses raw interval data and extracts features from the specified data type.
        (Handles potential DataProcessor import error)

        Args:
            interval_df_raw (pd.DataFrame): Raw sensor data for ONE interval/sample.
                                            Expected columns: ['acc_x', ..., 'dt', 'quat_x', ...].
            calibration_data (Dict): Loaded calibration data (biases, R_gravity_quat).

        Returns:
            Optional[Dict[str, Any]]: A dictionary of flattened features, or None if failed.
        """
        if DataProcessor is None:
             # Suppress repeated errors during LOOCV
             # print("Error: DataProcessor class not imported. Cannot reprocess data.")
             return None

        # 1. Reprocess the raw interval data using calibration
        reprocessed_data_dict = DataProcessor.reprocess_interval(interval_df_raw, calibration_data)

        if reprocessed_data_dict is None:
            # print("Error: Failed to reprocess interval data.") # Suppress verbosity
            return None

        # 2. Select the appropriate data based on self.feature_data_type
        accel_data_to_use: Optional[pd.DataFrame] = None
        gyro_data_to_use: Optional[pd.DataFrame] = None
        dt_series = reprocessed_data_dict.get('dt')
        if dt_series is None or dt_series.empty:
             # print("Error: 'dt' data missing after reprocessing.") # Suppress verbosity
             return None
        # Ensure dt is numpy array for feature functions
        dt_np = dt_series.values
        if dt_np.ndim > 1: # Ensure dt is 1D
             dt_np = dt_np.squeeze()

        # Select correct data source based on configuration
        source_map = {
            'raw': ('raw_acc', 'raw_gyro'),
            'corrected': ('corrected_acc', 'corrected_gyro'),
            'filtered': ('filtered_acc', 'filtered_gyro'),
            'rotated_gravity': ('rotated_gravity', 'filtered_gyro'), # Use filtered gyro for rotation features
            'rotated_current': ('rotated_current', 'filtered_gyro')  # Use filtered gyro for rotation features
        }
        acc_key, gyro_key = source_map.get(self.feature_data_type, (None, None))

        if acc_key: accel_data_to_use = reprocessed_data_dict.get(acc_key)
        if gyro_key: gyro_data_to_use = reprocessed_data_dict.get(gyro_key)


        # Validate that the selected data sources exist
        if accel_data_to_use is None:
            # print(f"Error: Could not retrieve specified accel data type '{self.feature_data_type}' after reprocessing.") # Suppress verbosity
            return None
        if gyro_data_to_use is None and gyro_key is not None :
             # print(f"Warning: Could not retrieve corresponding gyro data '{gyro_key}' for type '{self.feature_data_type}'. Gyro features might be missing.") # Suppress verbosity
             pass # Proceed, but gyro features might fail later.

        # Convert selected DataFrames to numpy arrays (3, N) for feature functions
        try:
            acc_np = accel_data_to_use.values.T # Transpose to get (3, N)
        except Exception as e:
             # print(f"Error converting accel data to numpy: {e}") # Suppress verbosity
             return None

        gyro_np = None
        if gyro_data_to_use is not None:
            try:
                gyro_np = gyro_data_to_use.values.T # Transpose to get (3, N)
            except Exception as e:
                # print(f"Warning: Error converting gyro data to numpy: {e}. Skipping gyro features.") # Suppress verbosity
                pass

        # 3. Extract features using DataProcessor static methods
        combined_features = {}
        acc_results = None
        gyro_results = None

        try:
            # Ensure dt_np is 1D array matching the number of samples (N)
            if acc_np.shape[1] == 0:
                 # print("Warning: Empty acceleration data array.") # Suppress verbosity
                 return None # Cannot extract features from empty data
            if acc_np.shape[1] != len(dt_np):
                 # print(f"Warning: Accel data ({acc_np.shape[1]} samples) and dt ({len(dt_np)} samples) length mismatch. Adjusting dt.") # Suppress verbosity
                 min_len = min(acc_np.shape[1], len(dt_np))
                 acc_np = acc_np[:, :min_len]
                 dt_np_acc = dt_np[:min_len]
            else:
                 dt_np_acc = dt_np
            if len(dt_np_acc) > 0: # Check if dt array is not empty
                acc_results = DataProcessor.process_acceleration(acc_np, dt_np_acc, zrc=self.zrc)
            else:
                # print("Warning: dt array for acceleration is empty after length adjustment.") # Suppress verbosity
                pass

        except ValueError as e:
             # print(f"Error in process_acceleration: {e}") # Suppress verbosity
             pass
        except Exception as e:
             # print(f"Unexpected error during acceleration feature extraction: {e}") # Suppress verbosity
             pass

        # Only process gyro if data exists and is valid
        if gyro_np is not None and gyro_np.size > 0:
            try:
                if gyro_np.shape[1] != len(dt_np):
                    # print(f"Warning: Gyro data ({gyro_np.shape[1]} samples) and dt ({len(dt_np)} samples) length mismatch. Adjusting dt.") # Suppress verbosity
                    min_len = min(gyro_np.shape[1], len(dt_np))
                    gyro_np = gyro_np[:, :min_len]
                    dt_np_gyro = dt_np[:min_len]
                else:
                    dt_np_gyro = dt_np

                if len(dt_np_gyro) > 0: # Check if dt array is not empty
                    gyro_results = DataProcessor.process_gyro(gyro_np, dt_np_gyro, zrc=self.zrc)
                else:
                    # print("Warning: dt array for gyroscope is empty after length adjustment.") # Suppress verbosity
                    pass
            except ValueError as e:
                # print(f"Error in process_gyro: {e}") # Suppress verbosity
                pass
            except Exception as e:
                # print(f"Unexpected error during gyroscope feature extraction: {e}") # Suppress verbosity
                pass

        # 4. Combine features safely
        if acc_results:
            _, acc_flat = acc_results
            if acc_flat: combined_features.update(acc_flat)
            # else: print(f"Warning: Accel feat extract returned empty for '{self.feature_data_type}'.")
        # else: print(f"Warning: Accel feat extract failed for '{self.feature_data_type}'.")

        if gyro_results:
            _, gyro_flat = gyro_results
            if gyro_flat: combined_features.update(gyro_flat)
            # else: print("Warning: Gyro feat extract returned empty dict.")

        if not combined_features:
             # print("Error: No features could be extracted for this interval.") # Suppress verbosity
             return None

        # Final check for non-finite values (redundant but safe)
        final_valid_features = {}
        nan_found = False
        for key, value in combined_features.items():
            if np.isfinite(value):
                final_valid_features[key] = value
            else:
                # print(f"Warning: Non-finite value detected for final feature '{key}'. Replacing with 0.") # Suppress verbosity
                final_valid_features[key] = 0.0
                nan_found = True
        # if nan_found: print("NaN replacement occurred in _extract_features") # Debug print

        return final_valid_features if final_valid_features else None


    def prepare_dataset(self,
                        interval_data_list: List[Tuple[pd.DataFrame, str, Dict, str]], # Added original_file_path
                        fit_scaler: bool = False,
                        verbose: bool = True # Control verbosity
                        ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare dataset by extracting features for each sensor data interval using calibration.

        Args:
            interval_data_list: List of tuples (interval_df_raw, label, calibration_data, original_file_path).
            fit_scaler (bool): If True, fit the StandardScaler. Set ONLY for training.
            verbose (bool): Whether to print detailed messages.

        Returns:
            Optional[Tuple[pd.DataFrame, pd.Series]]: Feature matrix X and labels y.
        """
        if not interval_data_list:
             if verbose: print("Error: Input interval_data_list is empty.")
             return None

        feature_dicts = []
        labels_list = []
        original_count = len(interval_data_list)
        processed_count = 0

        for i, (raw_df, label, calib_data, _) in enumerate(interval_data_list): # Ignore path here
            # Optional: Add progress for long preparations
            # if verbose and (i+1) % 100 == 0: print(f"  Processing interval {i+1}/{original_count}...")
            features = self._extract_features(raw_df, calib_data)
            if features is not None:
                 feature_dicts.append(features)
                 labels_list.append(label)
                 processed_count += 1
            # else: if verbose: print(f"  Skipping interval {i+1} due to extraction error.")

        if not feature_dicts:
            if verbose: print(f"Error: Feature extraction failed for ALL {original_count} intervals.")
            return None

        if processed_count < original_count and verbose:
            print(f"Warning: Skipped {original_count - processed_count} out of {original_count} intervals due to extraction errors.")

        X = pd.DataFrame(feature_dicts)
        y = pd.Series(labels_list, name="label")

        if X.empty:
             if verbose: print("Error: DataFrame is empty after feature extraction.")
             return None

        # --- Handle potential NaN/Inf columns and Imputation ---
        cols_before_nan_drop = set(X.columns)
        X = X.dropna(axis=1, how='all')
        cols_after_nan_drop = set(X.columns)
        dropped_cols = cols_before_nan_drop - cols_after_nan_drop
        if dropped_cols and verbose:
             print(f"Warning: Dropped {len(dropped_cols)} columns containing only NaN values.") #: {dropped_cols}") # Keep short
             if X.empty:
                  if verbose: print("Error: No valid feature columns remaining after dropping all-NaN columns.")
                  return None

        if X.isnull().values.any():
            nan_count = X.isnull().sum().sum()
            if verbose: print(f"Warning: Imputing {nan_count} NaN values using column median.")
            try:
                # Use try-except for median calculation resilience
                medians = X.median()
                X = X.fillna(medians)
                if X.isnull().values.any(): # Check if NaNs remain (e.g., median was NaN)
                     if verbose: print("Warning: NaNs still present after median imputation. Filling remaining with 0.")
                     X = X.fillna(0)
            except Exception as e:
                 if verbose: print(f"Error during median imputation: {e}. Filling all NaNs with 0.")
                 X = X.fillna(0) # Fallback to 0 if median calculation fails


        # --- Feature Selection ---
        available_columns = set(X.columns)
        if self._user_features:
            selected_features = [f for f in self._user_features if f in available_columns]
            missing = set(self._user_features) - available_columns
            if missing and verbose: print(f"Warning: Requested features not generated/available: {missing}")
            if not selected_features:
                 if verbose: print("Error: None of the user-specified features are available.")
                 return None
            if verbose: print(f"Using subset of {len(selected_features)} features specified by user.")
            X = X[selected_features]
            self._actual_features_used = selected_features
        else:
            self._actual_features_used = sorted(X.columns.tolist())
            X = X[self._actual_features_used]
            # if verbose: print(f"Using all {len(self._actual_features_used)} generated and valid features.") # Less verbose

        if X.empty or not self._actual_features_used:
            if verbose: print("Error: No valid features remaining after processing and selection.")
            return None

        # --- Scaling ---
        if fit_scaler:
            # if verbose: print("Fitting StandardScaler...")
            try:
                 self.scaler.fit(X)
                 # if verbose: print(f"Scaler fitted with {self.scaler.n_features_in_} features.")
            except ValueError as e: # Catch specific error for empty/NaN data during fit
                 if verbose: print(f"Error fitting scaler: {e}. Often due to NaN/inf in data.")
                 return None
            except Exception as e:
                 if verbose: print(f"Unexpected error fitting scaler: {e}")
                 return None

        try:
            X_scaled = self.scaler.transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=self._actual_features_used)
        except NotFittedError:
             # This should ideally be caught before calling predict, but good defense
             if verbose: print("Error: Scaler not fitted. Call train() first or use fit_scaler=True during preparation.")
             raise NotFittedError("Scaler must be fitted before transforming data.") # Re-raise for flow control
        except ValueError as e:
             if verbose:
                 print(f"Error during scaling transformation: {e}.")
                 print(f"Scaler expects {getattr(self.scaler, 'n_features_in_', 'N/A')} features (Names: {getattr(self.scaler, 'feature_names_in_', 'N/A')}).")
                 print(f"Input X has {X.shape[1]} features (Names: {X.columns.tolist()}).")
                 print("Check consistency between training and prediction features.")
             return None
        except Exception as e:
             if verbose: print(f"Unexpected error during scaling transformation: {e}")
             return None

        return X_scaled_df, y


    def train(self,
              interval_data_list: List[Tuple[pd.DataFrame, str, Dict, str]],
              perform_internal_eval: bool = True, # New flag
              test_size: float = 0.2,
              cv_folds: int = 5,
              verbose: bool = True): # Control verbosity
        """
        Trains the classifier. Optionally performs internal evaluation (split/CV).

        Args:
            interval_data_list: List of tuples (interval_df_raw, label, calibration_data, original_file_path).
            perform_internal_eval (bool): If True, perform train/test split or CV on the provided data.
                                          Set to False when using external CV like LOFO CV.
            test_size (float): Proportion for train/test split (only if perform_internal_eval=True).
            cv_folds (int): Number of folds for cross-validation (only if perform_internal_eval=True).
            verbose (bool): Whether to print detailed messages.
        """
        if verbose: print("Preparing dataset for training...")
        # Pass verbose flag to prepare_dataset
        prepared_data = self.prepare_dataset(interval_data_list, fit_scaler=True, verbose=verbose)

        if prepared_data is None:
            if verbose: print("Training aborted: Dataset preparation failed.")
            self.is_trained = False
            return self

        X_scaled, y = prepared_data
        self.classes_, _ = np.unique(y, return_inverse=True)
        if verbose:
            print(f"Training data shape: {X_scaled.shape}")
            print(f"Classes found: {self.classes_.tolist()}")
            print(f"Class distribution:\n{y.value_counts(normalize=True)}")

        # --- Evaluation Strategy (Optional Internal) ---
        if perform_internal_eval:
            if X_scaled.shape[0] < cv_folds * 2 and cv_folds > 1: # Need enough samples for folds
                 if verbose: print(f"Warning: Number of samples ({X_scaled.shape[0]}) is low for cv_folds ({cv_folds}). Reducing folds or skipping CV.")
                 cv_folds = max(2, X_scaled.shape[0] // 2) if X_scaled.shape[0] >= 4 else 0 # Ensure at least 2 samples per fold

            try:
                if cv_folds > 1:
                    if verbose: print(f"\nPerforming {cv_folds}-fold Cross-Validation (Internal)...")
                    temp_model_for_cv = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced', n_jobs=-1)
                    cv_scores = cross_val_score(temp_model_for_cv, X_scaled, y, cv=cv_folds, scoring='accuracy', error_score='raise')
                    if verbose: print(f"  Internal CV Accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

                if test_size > 0 and test_size < 1.0:
                     if verbose: print(f"\nPerforming single Train/Test Split (Internal, test_size={test_size})...")
                     min_class_count = y.value_counts().min()
                     n_splits_for_stratify = int(1 / test_size) if test_size > 0 else 1
                     stratify_param = y if min_class_count >= max(n_splits_for_stratify, 2) else None
                     if stratify_param is None and verbose:
                          print(f"Warning: The least populated class has only {min_class_count} samples. Stratification disabled for split.")

                     try:
                          X_train, X_test, y_train, y_test = train_test_split(
                               X_scaled, y, test_size=test_size, random_state=self.random_state, stratify=stratify_param
                          )
                          if verbose: print(f"  Training on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples.")

                          if verbose: print("  Fitting model on the training split...")
                          self.model.fit(X_train, y_train)
                          self.is_trained = True

                          y_pred = self.model.predict(X_test)
                          if verbose:
                              print("\n--- Classification Report (Internal Test Split) ---")
                              print(classification_report(y_test, y_pred, target_names=self.model.classes_.astype(str), zero_division=0))
                              print(f"Accuracy (Internal Test Split): {accuracy_score(y_test, y_pred):.4f}")
                              print("--------------------------------------------------")

                     except ValueError as e:
                          if verbose: print(f"\nError during Internal Train/Test Split: {e}. Training on full data instead.")
                          self.is_trained = False # Reset flag if split failed
                          # Fallback handled below

                # If no internal eval was done OR if split failed, train on full data
                if not self.is_trained:
                     if verbose: print("\nTraining final model on the entire provided dataset...")
                     self.model.fit(X_scaled, y)
                     self.is_trained = True
                     if verbose: print("Model trained on full provided data.")

            except Exception as fit_e:
                if verbose:
                    print(f"An unexpected error occurred during training/internal evaluation: {fit_e}")
                    import traceback
                    traceback.print_exc()
                self.is_trained = False

        else: # perform_internal_eval is False (e.g., during LOFO CV)
             if verbose: print("\nTraining model on the entire provided dataset (internal eval skipped)...")
             try:
                 self.model.fit(X_scaled, y)
                 self.is_trained = True
                 if verbose: print("Model trained.")
             except Exception as fit_e:
                 if verbose: print(f"An unexpected error occurred during training: {fit_e}")
                 self.is_trained = False

        # Store feature importances
        if self.is_trained and hasattr(self.model, 'feature_importances_'):
            try:
                 # Ensure features match
                 if self._actual_features_used and len(self._actual_features_used) == self.model.n_features_in_:
                      self.feature_importances_ = pd.Series(
                           self.model.feature_importances_,
                           index=self._actual_features_used
                      ).sort_values(ascending=False)
                      # if verbose: # Maybe only show this at the end of LOOCV?
                      #      print("\nTop 10 Feature Importances:")
                      #      print(self.feature_importances_.head(10))
                 elif hasattr(self.scaler, 'feature_names_in_') and len(self.scaler.feature_names_in_) == self.model.n_features_in_:
                      self.feature_importances_ = pd.Series(
                          self.model.feature_importances_,
                          index=self.scaler.feature_names_in_
                      ).sort_values(ascending=False)
                      # if verbose:
                      #      print("\nTop 10 Feature Importances (using scaler names):")
                      #      print(self.feature_importances_.head(10))
                 # else: if verbose: print("Warning: Could not align feature importances.")
            except Exception as fe_e:
                 if verbose: print(f"Warning: Error storing feature importances: {fe_e}")

        return self


    # Predict for a single interval - unchanged
    def predict(self,
                sensor_data_raw: pd.DataFrame,
                calibration_data: Dict
                ) -> Optional[Tuple[str, float]]:
        """ Predict street texture for a SINGLE raw sensor data interval. """
        if not self.is_trained:
            # Suppress error during prediction loops, check should be done before calling
            # raise NotFittedError("Model must be trained before prediction.")
            return None
        if self._actual_features_used is None:
             # raise ValueError("Model state inconsistent: actual features used are unknown.")
             return None
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
             # raise NotFittedError("Scaler has not been fitted.")
             return None

        # 1. Extract features
        features_dict = self._extract_features(sensor_data_raw, calibration_data)
        if features_dict is None:
            # print("Prediction failed: Feature extraction error.") # Suppress verbosity
            return None

        # 2. Prepare feature vector (single row DataFrame)
        try:
             pred_series = pd.Series(features_dict)
             pred_series_reindexed = pred_series.reindex(self._actual_features_used) # Fills missing with NaN
             X_pred = pd.DataFrame([pred_series_reindexed], columns=self._actual_features_used)

             # Check for NaNs *after* reindexing
             if X_pred.isnull().values.any():
                  nan_cols = X_pred.columns[X_pred.isnull().any()].tolist()
                  # print(f"Warning: NaN values detected in prediction features: {nan_cols}. Imputing with 0.") # Suppress verbosity
                  # Simple imputation with 0 for prediction time. Could use stored means/medians from scaler if available.
                  X_pred = X_pred.fillna(0)

             # Re-check for missing features critical to model (should have been caught by reindex + fillna)
             # missing_features = X_pred.columns[X_pred.isna().all()].tolist() # This check is less likely needed after fillna(0)
             # if missing_features:
             #      print(f"Prediction failed: Critical features missing even after imputation: {missing_features}")
             #      return None

        except Exception as e:
             # print(f"Error creating prediction DataFrame: {e}") # Suppress verbosity
             return None

        # 3. Scale features
        try:
            if list(X_pred.columns) != self._actual_features_used:
                 # print("Error: Prediction feature columns mismatch. Reordering.") # Suppress verbosity
                 try:
                      X_pred = X_pred[self._actual_features_used]
                 except KeyError as ke:
                      # print(f"Fatal Error: Cannot reorder, key error: {ke}") # Suppress verbosity
                      return None

            X_scaled_np = self.scaler.transform(X_pred)
            X_scaled_df = pd.DataFrame(X_scaled_np, columns=self._actual_features_used)
        except NotFittedError:
             # print("Error: Scaler not fitted, cannot predict.") # Suppress verbosity
             return None # Should be caught before calling predict loop
        except ValueError as e:
             # print(f"Error during prediction scaling: {e}") # Suppress verbosity
             # print(f" Scaler: {getattr(self.scaler, 'n_features_in_', 'N/A')} features. Input: {X_pred.shape[1]}.")
             return None
        except Exception as e:
             # print(f"Unexpected error during prediction scaling: {e}") # Suppress verbosity
             return None

        # 4. Predict probabilities
        try:
            probas = self.model.predict_proba(X_scaled_df)
            if probas is None or probas.shape[0] == 0:
                 # print("Error: predict_proba returned empty result.") # Suppress verbosity
                 return None
        except Exception as e:
            # print(f"Error during probability prediction: {e}") # Suppress verbosity
            return None

        # 5. Get predicted label and confidence
        try:
             predicted_index = np.argmax(probas[0])
             if not hasattr(self.model, 'classes_') or self.model.classes_ is None:
                  # print("Error: Model classes not available.") # Suppress verbosity
                  return None
             if predicted_index >= len(self.model.classes_):
                  # print(f"Error: Predicted index out of bounds.") # Suppress verbosity
                  return None

             predicted_label = self.model.classes_[predicted_index]
             confidence = probas[0][predicted_index]
             return predicted_label, confidence
        except IndexError as e:
             # print(f"Error accessing prediction results: {e}") # Suppress verbosity
             return None
        except Exception as e:
             # print(f"Unexpected error getting prediction label/confidence: {e}") # Suppress verbosity
             return None

    # predict_from_file: Keep this method as is for predicting *new*, single files
    # after a final model has been trained. We won't use it *inside* the LOFO CV loop.
    def predict_from_file(self,
                          raw_csv_path: str,
                          interval_types: List[str] = ['time', 'sample', 'event'] # Which interval types to use
                         ) -> Optional[Tuple[str, float, Dict[str, int]]]:
        """
        Predicts the classification for an entire raw data file using a pre-trained model.
        (Loads file, gets calibration, generates intervals, predicts, aggregates)

        Args:
            raw_csv_path (str): Path to the raw sensor data CSV file (11 columns).
            interval_types (List[str]): Which types of intervals generated by
                                        DataProcessor.create_intervals to use for prediction.
                                        Options: 'time', 'sample', 'event'. Default uses all.

        Returns:
            Optional[Tuple[str, float, Dict[str, int]]]:
                - Winning predicted label (string).
                - Average confidence score for the winning label (float).
                - Dictionary of vote counts for each label.
            Returns None if prediction fails.
        """
        print(f"\n--- Starting Prediction for File: {os.path.basename(raw_csv_path)} ---")
        if not self.is_trained:
            print("Error: Model must be trained before predicting.")
            return None
        if self._actual_features_used is None:
             print("Error: Model state inconsistent: actual features used are unknown (train first).")
             return None
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
             print("Error: Scaler has not been fitted (train first).")
             return None
        if DataProcessor is None:
             print("Error: DataProcessor not available for interval generation.")
             return None

        # --- 1. Find and Load Calibration Data ---
        json_path = os.path.splitext(raw_csv_path)[0] + '.json'
        if not os.path.exists(json_path):
            print(f"Error: Calibration file not found at {json_path}")
            return None
        try:
            with open(json_path, 'r') as f:
                calibration_data = json.load(f)
            if not all(k in calibration_data for k in ['accel_bias', 'gyro_bias', 'R_gravity_quat']):
                 print(f"Error: Calibration JSON {json_path} missing required keys.")
                 return None
            # print(f"Loaded calibration data from {json_path}")
        except Exception as e:
            print(f"Error reading/decoding calibration file {json_path}: {e}")
            return None

        # --- 2. Load Raw Sensor Data ---
        column_names_raw = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'dt',
                            'quat_x', 'quat_y', 'quat_z', 'quat_w']
        try:
            full_trip_df = pd.read_csv(raw_csv_path, header=None, names=column_names_raw)
            if full_trip_df.empty or len(full_trip_df.columns) != 11:
                print(f"Error: Raw data file {raw_csv_path} is empty or has incorrect format.")
                return None
            # print(f"Loaded raw data ({len(full_trip_df)} samples) from {raw_csv_path}")
        except Exception as e:
            print(f"Error loading raw data CSV {raw_csv_path}: {e}")
            return None

        # --- 3. Generate Intervals ---
        try:
            raw_data_list = full_trip_df.values.tolist()
            interval_data_dict = DataProcessor.create_intervals(raw_data_list)
        except Exception as e:
             print(f"Error during interval generation: {e}")
             return None

        if not interval_data_dict:
            print("Error: No intervals could be generated from the data.")
            return None

        intervals_to_predict: List[pd.DataFrame] = []
        for interval_type in interval_types:
             if interval_type in interval_data_dict:
                  for interval_list in interval_data_dict[interval_type]:
                       if interval_list and len(interval_list) > 1:
                            interval_df = pd.DataFrame(interval_list, columns=column_names_raw)
                            intervals_to_predict.append(interval_df)

        if not intervals_to_predict:
            print(f"Error: No valid intervals generated for types {interval_types}. Cannot predict.")
            return None
        # print(f"Generated {len(intervals_to_predict)} intervals for prediction.")

        # --- 4. Predict on Each Interval ---
        all_predictions = [] # Store tuples of (label, confidence)
        for i, interval_df in enumerate(intervals_to_predict):
            prediction_result = self.predict(interval_df, calibration_data) # Use self.predict here
            if prediction_result:
                all_predictions.append(prediction_result)

        if not all_predictions:
            print("Error: Prediction failed for all generated intervals.")
            return None
        # print(f"Successfully predicted on {len(all_predictions)} out of {len(intervals_to_predict)} intervals.")

        # --- 5. Aggregate Results (Majority Voting) ---
        predicted_labels = [label for label, conf in all_predictions]
        label_counts = Counter(predicted_labels)
        if not label_counts: # Should not happen if all_predictions is not empty
             print("Internal Error: No labels found after successful predictions.")
             return None

        winning_label, win_count = label_counts.most_common(1)[0]
        winning_confidences = [conf for label, conf in all_predictions if label == winning_label]
        average_confidence = sum(winning_confidences) / len(winning_confidences) if winning_confidences else 0.0

        print(f"  Interval Prediction Counts: {dict(label_counts)}")
        print(f"  Majority Vote Winner: '{winning_label}' ({win_count} votes)")
        print(f"  Average Confidence for '{winning_label}': {average_confidence:.4f}")
        print("------------------------------------")

        return winning_label, average_confidence, dict(label_counts)



# --- MODIFIED Helper function to prepare data (adds original file path) ---
def prepare_training_data_from_files(
    raw_file_paths: List[str],
    raw_file_labels: List[str]
) -> List[Tuple[pd.DataFrame, str, Dict, str]]: # Added original_file_path as 4th element
    """
    Processes raw data files, loads calibration, generates intervals,
    and returns data structured for the StreetTextureClassifier, including the source file path.

    Args:
        raw_file_paths: List of paths to the raw data CSV files.
        raw_file_labels: List of labels corresponding to each raw file.

    Returns:
        List[Tuple[pd.DataFrame, str, Dict, str]]:
            List where each element is (interval_df, label, calibration_data, original_file_path).
            Returns empty list if processing fails.
    """
    if DataProcessor is None:
         print("Error: Cannot prepare training data - DataProcessor unavailable.")
         return []

    all_interval_tuples = []
    column_names_raw = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'dt',
                        'quat_x', 'quat_y', 'quat_z', 'quat_w']

    if len(raw_file_paths) != len(raw_file_labels):
         print("Error: Mismatch between number of file paths and labels.")
         return []

    print("\n--- Preparing Intervals from Files ---")
    for file_path_csv, file_label in zip(raw_file_paths, raw_file_labels):
        # print(f"Processing: {os.path.basename(file_path_csv)} ({file_label})...") # Less verbose

        file_path_json = os.path.splitext(file_path_csv)[0] + '.json'
        calibration_data = None
        if not os.path.exists(file_path_json):
            print(f"  Warning: Calibration JSON not found: {file_path_json}. Skipping file.")
            continue
        try:
            with open(file_path_json, 'r') as f:
                calibration_data = json.load(f)
            if not all(k in calibration_data for k in ['accel_bias', 'gyro_bias', 'R_gravity_quat']):
                 print(f"  Warning: Calibration JSON missing keys in {file_path_json}. Skipping file.")
                 continue
        except Exception as e:
            print(f"  Warning: Error reading calibration {file_path_json}: {e}. Skipping file.")
            continue

        try:
            interval_df_full_trip = pd.read_csv(file_path_csv, header=None, names=column_names_raw)
            if interval_df_full_trip.empty or len(interval_df_full_trip.columns) != 11:
                 print(f"  Warning: Raw data CSV empty or wrong format: {file_path_csv}. Skipping.")
                 continue
            if interval_df_full_trip.isnull().values.any():
                 nan_frac = interval_df_full_trip.isnull().sum().sum() / interval_df_full_trip.size
                 if nan_frac > 0.2: # Allow up to 20% NaNs? Adjust threshold.
                      print(f"  Warning: Raw data has >{nan_frac:.1%} NaN values in {file_path_csv}. Skipping.")
                      continue
                 # else: print(f"  Warning: Raw data contains NaNs ({nan_frac:.1%}) in {file_path_csv}. Proceeding.")

            raw_data_list_of_lists = interval_df_full_trip.values.tolist()
            # Generate standard intervals (time, sample, event)
            interval_data_dict = DataProcessor.create_intervals(raw_data_list_of_lists)

            generated_count = 0
            min_interval_len = 10 # Minimum samples for a usable interval
            if interval_data_dict:
                for interval_type, intervals_list in interval_data_dict.items():
                    for interval_list in intervals_list:
                        if not interval_list or len(interval_list) < min_interval_len:
                            continue
                        interval_df_sub = pd.DataFrame(interval_list, columns=column_names_raw)
                        # Append tuple: (DataFrame, label, calibration, source_file_path)
                        all_interval_tuples.append((interval_df_sub, file_label, calibration_data, file_path_csv))
                        generated_count += 1
                # print(f"  Generated {generated_count} sub-intervals.")
            else:
                 print(f"  Warning: No sub-intervals generated by create_intervals for {file_path_csv}.")
                 # Fallback: Use full trip if long enough?
                 if len(interval_df_full_trip) >= min_interval_len:
                     print("  Using full trip as a single interval as fallback.")
                     all_interval_tuples.append((interval_df_full_trip, file_label, calibration_data, file_path_csv))
                     generated_count = 1

            if generated_count == 0:
                print(f"  Warning: No usable intervals found for file: {os.path.basename(file_path_csv)}")


        except FileNotFoundError:
            print(f"  Warning: Raw data file not found: {file_path_csv}. Skipping.")
        except pd.errors.EmptyDataError:
             print(f"  Warning: Raw data file is empty (pandas error): {file_path_csv}. Skipping.")
        except Exception as e:
            print(f"  Warning: Error processing file {file_path_csv}: {e}")
            # import traceback # Optional detailed traceback for debugging
            # traceback.print_exc()

    print(f"--- Total intervals prepared from all files: {len(all_interval_tuples)} ---")
    if not all_interval_tuples and raw_file_paths:
         print("Warning: No valid intervals could be prepared from the provided files.")

    return all_interval_tuples


# --- Main Execution Block with Leave-One-File-Out CV ---
if __name__ == "__main__":
     # --- Setup Paths (Adjust as needed) ---
     try:
         script_dir = os.path.dirname(__file__)
         project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
     except NameError: # Handle case where __file__ is not defined (e.g., interactive)
         project_root = os.path.abspath('.') # Assume current dir is project root

     data_dir = os.path.join(project_root, 'data', 'raw')

     print(f"Project Root (estimated): {project_root}")
     print(f"Data Directory: {data_dir}")

     if not os.path.isdir(data_dir):
          print(f"Error: Data directory not found at {data_dir}.")
          exit()
     if DataProcessor is None:
          print("Error: DataProcessor class is not available. Cannot proceed.")
          exit()

     # --- 1. Define Data Files and Labels ---
     # Use the same file lists as before
     stadt_files = [os.path.join(data_dir, f) for f in ['data_stadt.csv', 'data_stadt_1.csv', 'data_stadt_2.csv', 'data_stadt_3.csv', 'data_stadt_4.csv']]
     landstrasse_files = [os.path.join(data_dir, f) for f in ['data_landstraße.csv', 'data_landstraße_1.csv', 'data_landstraße_2.csv', 'data_landstraße_3.csv', 'data_landstraße_4.csv', 'data_landstraße_5.csv']]
     autobahn_files = [os.path.join(data_dir, f) for f in ['data_autobahn.csv', 'data_autobahn_1.csv']]
     dorf_files = [os.path.join(data_dir, f) for f in ['data_dorf.csv', 'data_dorf_1.csv']]

     all_files = stadt_files + landstrasse_files + autobahn_files + dorf_files
     all_labels = ["stadt"] * len(stadt_files) + ["landstraße"] * len(landstrasse_files) + ["autobahn"] * len(autobahn_files) + ["dorf"] * len(dorf_files)

     # --- Data Validation ---
     print("\nChecking data files:")
     valid_files = []
     valid_labels = []
     for f, l in zip(all_files, all_labels):
          json_f = os.path.splitext(f)[0] + '.json'
          if os.path.exists(f) and os.path.exists(json_f):
               # print(f"  [OK] {os.path.basename(f)}") # Less verbose
               valid_files.append(f)
               valid_labels.append(l)
          else:
               print(f"  [MISSING!] CSV or JSON for: {os.path.basename(f)}")

     if not valid_files:
          print(f"\nError: No valid data files (CSV+JSON pairs) found. Exiting.")
          exit()
     if len(valid_files) < len(all_files):
          print("\nWarning: Some files were missing. Proceeding with available files.")

     unique_files = sorted(list(set(valid_files))) # Get unique file paths
     num_unique_files = len(unique_files)
     print(f"\nFound {num_unique_files} unique valid files for LOFO CV.")
     if num_unique_files < 2:
          print("Error: Need at least 2 unique files for Leave-One-File-Out evaluation. Exiting.")
          exit()

     # Prepare all intervals ONCE
     all_prepared_intervals = prepare_training_data_from_files(valid_files, valid_labels)

     if not all_prepared_intervals:
          print("\nError: Could not prepare any intervals from the valid files. Exiting.")
          exit()

     # Check class distribution among intervals
     interval_label_counts = Counter([label for _, label, _, _ in all_prepared_intervals])
     print(f"Total interval distribution by class: {dict(interval_label_counts)}")
     if len(interval_label_counts) < 2:
          print("Error: Only one class represented in the prepared intervals. Cannot train/evaluate.")
          exit()

     # --- 2. Perform Leave-One-File-Out Cross-Validation ---
     print(f"\n--- Starting Leave-One-File-Out Cross-Validation ({num_unique_files} folds) ---")
     feature_type = 'rotated_gravity' # Or choose another: 'raw', 'corrected', 'filtered', 'rotated_current'
     print(f"Using feature type: {feature_type}")

     lofo_results = [] # Store results for each held-out file
     all_labels_true = [] # Store true labels for overall metrics
     all_labels_pred = [] # Store predicted labels for overall metrics
     fold_feature_importances = [] # Store importances from each fold

     start_time_lofo = time.time()

     for i, file_to_hold_out in enumerate(unique_files):
         fold_start_time = time.time()
         print(f"\n[Fold {i+1}/{num_unique_files}] Holding out: {os.path.basename(file_to_hold_out)}")

         # Split intervals into train and test sets for this fold
         train_intervals_fold = [t for t in all_prepared_intervals if t[3] != file_to_hold_out]
         test_intervals_fold = [t for t in all_prepared_intervals if t[3] == file_to_hold_out]

         if not test_intervals_fold:
             print("  Warning: No intervals found for the held-out file. Skipping fold.")
             continue
         if not train_intervals_fold:
             print("  Warning: No training intervals found when holding out this file. Skipping fold.")
             continue

         # Get the true label for the held-out file (all its intervals share the same label)
         true_label_held_out = test_intervals_fold[0][1]
         print(f"  True label for held-out file: '{true_label_held_out}'")
         print(f"  Training intervals: {len(train_intervals_fold)}, Test intervals: {len(test_intervals_fold)}")

         # Instantiate a FRESH classifier for this fold
         classifier_fold = StreetTextureClassifier(
             features=None,
             feature_data_type=feature_type,
             random_state=42 # Use same random state for model consistency across folds
         )

         # Train the classifier on the training intervals for this fold
         # Set perform_internal_eval=False, verbose=False to reduce output noise
         classifier_fold.train(train_intervals_fold, perform_internal_eval=False, verbose=False)

         if not classifier_fold.is_trained:
              print("  Error: Model training failed for this fold. Skipping prediction.")
              continue

         # Store feature importances for this fold if available
         if classifier_fold.feature_importances_ is not None:
              fold_feature_importances.append(classifier_fold.feature_importances_)

         # Predict on each interval of the held-out file
         fold_predictions = [] # List of (predicted_label, confidence)
         for interval_df, _, calib_data, _ in test_intervals_fold:
             pred_result = classifier_fold.predict(interval_df, calib_data)
             if pred_result:
                 fold_predictions.append(pred_result)
             # else: print("   -> Interval prediction failed") # Very verbose

         if not fold_predictions:
             print("  Error: Prediction failed for all intervals in the held-out file.")
             # Record failure? Or just skip? Let's record a placeholder.
             predicted_label_majority = "[Prediction Failed]"
             avg_confidence_majority = 0.0
             interval_votes = {}
         else:
             # Aggregate results for the held-out file (Majority Voting)
             fold_pred_labels = [label for label, conf in fold_predictions]
             interval_votes = Counter(fold_pred_labels)
             predicted_label_majority, win_count = interval_votes.most_common(1)[0]
             winning_confidences = [conf for label, conf in fold_predictions if label == predicted_label_majority]
             avg_confidence_majority = sum(winning_confidences) / len(winning_confidences) if winning_confidences else 0.0

             print(f"  Interval votes: {dict(interval_votes)}")
             print(f"  --> Fold Prediction (Majority): '{predicted_label_majority}' (Confidence: {avg_confidence_majority:.4f})")

         # Store results for this fold
         fold_result_data = {
             'file': os.path.basename(file_to_hold_out),
             'true_label': true_label_held_out,
             'predicted_label': predicted_label_majority,
             'avg_confidence': avg_confidence_majority,
             'interval_votes': dict(interval_votes),
             'correct': predicted_label_majority == true_label_held_out if predicted_label_majority != "[Prediction Failed]" else False
         }
         lofo_results.append(fold_result_data)

         # Append to overall lists for final metrics (only if prediction succeeded)
         if predicted_label_majority != "[Prediction Failed]":
             all_labels_true.append(true_label_held_out)
             all_labels_pred.append(predicted_label_majority)

         fold_duration = time.time() - fold_start_time
         print(f"  Fold completed in {fold_duration:.2f} seconds.")


     total_lofo_duration = time.time() - start_time_lofo
     print(f"\n--- LOFO Cross-Validation Completed in {total_lofo_duration:.2f} seconds ---")

     # --- 3. Analyze LOFO CV Results ---
     if not lofo_results:
         print("No folds were successfully completed.")
     else:
         num_correct = sum(r['correct'] for r in lofo_results)
         num_total_evaluated = len(all_labels_true) # Only count folds where prediction succeeded
         overall_accuracy = num_correct / num_total_evaluated if num_total_evaluated > 0 else 0.0

         print("\n--- Overall LOFO CV Results ---")
         print(f"Total Files Evaluated: {num_total_evaluated}/{num_unique_files}")
         print(f"Correctly Classified Files: {num_correct}")
         print(f"Overall Accuracy (File Level): {overall_accuracy:.4f}")

         # Detailed results per file
         print("\n--- Results per Fold (File) ---")
         for r in lofo_results:
             status = "Correct" if r['correct'] else ("Incorrect" if r['predicted_label'] != "[Prediction Failed]" else "Failed")
             print(f"  File: {r['file']:<25} | True: {r['true_label']:<12} | Pred: {r['predicted_label']:<18} | Conf: {r['avg_confidence']:.3f} | Status: {status}")

         # Overall Confusion Matrix (File Level)
         if num_total_evaluated > 0:
             unique_classes_overall = sorted(list(set(all_labels_true) | set(all_labels_pred)))
             print("\n--- Overall Confusion Matrix (File Level) ---")
             try:
                 cm = confusion_matrix(all_labels_true, all_labels_pred, labels=unique_classes_overall)
                 print(f"Labels: {unique_classes_overall}")
                 print(cm)

                 # Optional: Plot confusion matrix
                 try:
                     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes_overall)
                     disp.plot(cmap=plt.cm.Blues)
                     plt.title("LOFO CV Confusion Matrix (File Level)")
                     plt.xticks(rotation=45, ha='right')
                     plt.tight_layout()
                     # Save or show the plot
                     cm_filename = f"lofo_confusion_matrix_{feature_type}.png"
                     plt.savefig(cm_filename)
                     print(f"Confusion matrix plot saved to {cm_filename}")
                     # plt.show() # Uncomment to display interactively
                     plt.close() # Close the plot window
                 except Exception as plot_e:
                     print(f"Warning: Could not plot confusion matrix: {plot_e}")

             except Exception as cm_e:
                 print(f"Error calculating confusion matrix: {cm_e}")


         # Average Feature Importances across folds
         if fold_feature_importances:
              try:
                   avg_importances = pd.concat(fold_feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)
                   print("\n--- Average Feature Importances (across LOFO folds) ---")
                   print(avg_importances.head(15)) # Show top 15
              except Exception as fi_e:
                   print(f"\nWarning: Could not average feature importances: {fi_e}")


     # --- 4. Optional: Train Final Model on ALL Data ---
     train_final_model = True # Set to False if you only want the evaluation
     if train_final_model and all_prepared_intervals:
         print("\n--- Training Final Model on ALL Data ---")
         final_classifier = StreetTextureClassifier(
             features=None,
             feature_data_type=feature_type,
             random_state=42
         )
         # Use verbose=True here for final training output
         final_classifier.train(all_prepared_intervals, perform_internal_eval=True, test_size=0, cv_folds=0, verbose=True)

         if final_classifier.is_trained:
             print("Final model trained successfully on all data.")
             # You could now save this final_classifier instance (e.g., using pickle or joblib)
             # Or use it for prediction on new, unseen files using final_classifier.predict_from_file(...)
             # Example:
             # import joblib
             # model_filename = f'street_texture_classifier_{feature_type}_final.joblib'
             # joblib.dump(final_classifier, model_filename)
             # print(f"Final model saved to {model_filename}")

             # Example prediction with the final model:
             # if len(valid_files) > 0:
             #    test_file = valid_files[0] # Predict one of the training files again (just for demo)
             #    print(f"\n--- Example Prediction using Final Model on {os.path.basename(test_file)} ---")
             #    final_classifier.predict_from_file(test_file)

         else:
             print("Failed to train the final model on all data.")

     print("\n--- End of Script ---")
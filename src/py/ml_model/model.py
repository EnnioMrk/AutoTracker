from typing import List, Any

import numpy as np
import pandas as pd
from numpy import ndarray, dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class StreetTextureClassifier:
    def __init__(self,
                 features: List[str] = None,
                 model_type: str = 'random_forest',
                 random_state: int = 42):
        """
        Flexible street texture classification model.

        Args:
            features (List[str]): List of feature names to use in classification
            model_type (str): Type of classifier to use
            random_state (int): Seed for reproducibility
        """
        # Default feature names if not provided
        self._default_features = [
            'acc_mean_x', 'acc_mean_y', 'acc_mean_z',
            'acc_median_x', 'acc_median_y', 'acc_median_z',
            'acc_max_x', 'acc_max_y', 'acc_max_z',
            'acc_min_x', 'acc_min_y', 'acc_min_z',
            'acc_energy', 'acc_std',
            'gyro_mean_x', 'gyro_mean_y', 'gyro_mean_z',
            'gyro_median_x', 'gyro_median_y', 'gyro_median_z',
            'gyro_max_x', 'gyro_max_y', 'gyro_max_z',
            'gyro_min_x', 'gyro_min_y', 'gyro_min_z',
            'gyro_energy', 'gyro_std'
        ]

        # Use provided features or default
        self.features = features or self._default_features

        # Initialize scaler and model
        self.scaler = StandardScaler()

        # Select model type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Training tracking
        self.is_trained = False
        self.feature_importances_ = None

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract specified features from raw sensor data.

        Args:
            data (pd.DataFrame): Raw sensor data

        Returns:
            pd.DataFrame: Extracted features
        """
        # Placeholder for feature extraction logic
        # This method should be customized based on your specific data collection
        features_dict = {}

        # Acceleration features
        acc_columns = [col for col in data.columns if col.startswith('acc')]
        features_dict.update({
            'acc_mean_x': data['acc_x'].mean(),
            'acc_mean_y': data['acc_y'].mean(),
            'acc_mean_z': data['acc_z'].mean(),
            'acc_median_x': data['acc_x'].median(),
            # Add more acceleration features...
        })

        # Gyroscope features
        gyro_columns = [col for col in data.columns if col.startswith('gyro')]
        features_dict.update({
            'gyro_mean_x': data['gyro_x'].mean(),
            'gyro_mean_y': data['gyro_y'].mean(),
            'gyro_mean_z': data['gyro_z'].mean(),
            # Add more gyroscope features...
        })

        # Energy calculations
        features_dict['acc_energy'] = np.sum(data[acc_columns] ** 2)
        features_dict['gyro_energy'] = np.sum(data[gyro_columns] ** 2)

        # Standard deviation
        features_dict['acc_std'] = data[acc_columns].std().mean()
        features_dict['gyro_std'] = data[gyro_columns].std().mean()

        return pd.DataFrame([features_dict])

    def prepare_dataset(self,
                        sensor_data: List[pd.DataFrame],
                        labels: List[str]) -> tuple:
        """
        Prepare dataset by extracting features from sensor data.

        Args:
            sensor_data (List[pd.DataFrame]): List of sensor data for each sample
            labels (List[str]): Corresponding texture labels

        Returns:
            tuple: X (features), y (labels)
        """
        # Extract features for each dataset
        feature_matrices = [self._extract_features(data) for data in sensor_data]
        X = pd.concat(feature_matrices, ignore_index=True)

        # Validate input
        assert len(X) == len(labels), "Number of feature sets must match number of labels"

        return X, labels

    def train(self,
              sensor_data: List[pd.DataFrame],
              labels: List[str],
              test_size: float = 0.2):
        """
        Train the street texture classifier.

        Args:
            sensor_data (List[pd.DataFrame]): List of sensor data for each sample
            labels (List[str]): Corresponding texture labels
            test_size (float): Proportion of data to use for testing
        """
        # Prepare dataset
        X, y = self.prepare_dataset(sensor_data, labels)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Store feature importances
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        self.is_trained = True
        return self

    def predict(self, sensor_data: pd.DataFrame):
        """
        Predict street texture for new sensor data and return the predicted label with confidence.

        Args:
            sensor_data (pd.DataFrame): Sensor data for prediction

        Returns:
            str: Predicted texture label
            float: Confidence level (probability)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Extract features
        X = self._extract_features(sensor_data)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        probas = self.model.predict_proba(X_scaled)

        # Get the predicted label (class with the highest probability)
        predicted_label = self.model.classes_[np.argmax(probas)]

        # Get the confidence (probability of the predicted class)
        confidence = np.max(probas)

        return predicted_label, confidence

    def add_features(self, new_features: List[str]):
        """
        Add new features to the model's feature list.

        Args:
            new_features (List[str]): List of new feature names to add
        """
        self.features.extend(new_features)
        print(f"Added features: {new_features}")

    def remove_features(self, features_to_remove: List[str]):
        """
        Remove specified features from the model's feature list.

        Args:
            features_to_remove (List[str]): List of feature names to remove
        """
        for feature in features_to_remove:
            if feature in self.features:
                self.features.remove(feature)
        print(f"Removed features: {features_to_remove}")
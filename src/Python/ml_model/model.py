import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and prepare your data
def prepare_data(file_paths, labels):
    features = []
    targets = []

    for file_path, label in zip(file_paths, labels):
        # Load data
        data = pd.read_csv(file_path, header=None,
                           names=['hz', 'z_acc', 'x_acc', 'y_acc'])

        # Process in windows (e.g., 1-second windows)
        window_size = 50  # Adjust based on your sampling rate

        for i in range(0, len(data) - window_size, int(window_size / 2)):
            window = data.iloc[i:i + window_size]

            # Extract features from this window
            window_features = extract_features(window)
            features.append(window_features)
            targets.append(label)

    return np.array(features), np.array(targets)


def extract_features(window):
    # Time domain features
    mean_x = window['x_acc'].mean()
    mean_y = window['y_acc'].mean()
    mean_z = window['z_acc'].mean()
    std_x = window['x_acc'].std()
    std_y = window['y_acc'].std()
    std_z = window['z_acc'].std()

    # Calculate FFT for frequency domain features
    fft_x = np.abs(np.fft.fft(window['x_acc']))
    fft_y = np.abs(np.fft.fft(window['y_acc']))
    fft_z = np.abs(np.fft.fft(window['z_acc']))

    # Most dominant frequencies
    dom_freq_x = np.argmax(fft_x[1:len(fft_x) // 2]) + 1
    dom_freq_y = np.argmax(fft_y[1:len(fft_y) // 2]) + 1
    dom_freq_z = np.argmax(fft_z[1:len(fft_z) // 2]) + 1

    # Calculate energy
    energy_x = np.sum(fft_x ** 2) / len(fft_x)
    energy_y = np.sum(fft_y ** 2) / len(fft_y)
    energy_z = np.sum(fft_z ** 2) / len(fft_z)

    return [mean_x, mean_y, mean_z, std_x, std_y, std_z,
            dom_freq_x, dom_freq_y, dom_freq_z,
            energy_x, energy_y, energy_z]


# Train the model
X, y = prepare_data(['highway_data.csv', 'city_data.csv', 'gravel_data.csv'],
                    ['highway', 'city', 'gravel'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'road_classifier.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

# Evaluate
accuracy = model.score(X_test_scaled, y_test)
print(f"Model accuracy: {accuracy:.2f}")
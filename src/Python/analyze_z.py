import numpy as np
from scipy.integrate import simpson
import json
from collections import defaultdict

def compute_integral_simpson(interval, sampling_rate=250):
    dt = 1 / sampling_rate
    return simpson(interval, dx=dt)

def read_file(file):
    with open(file, "r") as f:
        return [float(line.strip()) for line in f]

def process_z_values(z_values, sampling_rate=250):
    if not z_values:
        return None

    dt = 1 / sampling_rate
    intervals = []
    current_interval = [z_values[0]]

    # Calculate overall statistics
    std_dev = np.std(z_values)
    mean = np.mean(z_values)
    positive_deviation = sum(a - mean for a in z_values if a > mean)
    negative_deviation = sum(mean - a for a in z_values if a < mean)
    ratio = positive_deviation / negative_deviation if negative_deviation != 0 else float('inf')

    # Split into intervals
    for z in z_values[1:]:
        if (z > 1 and current_interval[0] > 1) or (z < 1 and current_interval[0] < 1):
            current_interval.append(z)
        else:
            intervals.append(current_interval)
            current_interval = [z]
    if current_interval:
        intervals.append(current_interval)

    # Separate positive and negative intervals
    positive_intervals = [intv for intv in intervals if intv[0] > 1]
    negative_intervals = [intv for intv in intervals if intv[0] < 1]

    # Calculate features
    num_positive = len(positive_intervals)
    num_negative = len(negative_intervals)

    features = {
        'num_intervals': len(intervals),
        'num_positive_intervals': num_positive,
        'num_negative_intervals': num_negative,
        'mean_positive_extreme': np.mean([max(intv) for intv in positive_intervals]) if num_positive > 0 else 0,
        'mean_negative_extreme': np.mean([min(intv) for intv in negative_intervals]) if num_negative > 0 else 0,
        'mean_duration_positive': np.mean([len(intv)*dt for intv in positive_intervals]) if num_positive > 0 else 0,
        'mean_duration_negative': np.mean([len(intv)*dt for intv in negative_intervals]) if num_negative > 0 else 0,
        'mean_integral_positive': np.mean([compute_integral_simpson(intv, sampling_rate) for intv in positive_intervals]) if num_positive > 0 else 0,
        'mean_integral_negative': np.mean([compute_integral_simpson(intv, sampling_rate) for intv in negative_intervals]) if num_negative > 0 else 0,
        'deviation_ratio': ratio,
        'overall_std_dev': std_dev
    }
    return features

def process_file(file_path, sampling_rate=250):
    z_values = read_file(file_path)
    return process_z_values(z_values, sampling_rate)

def save_features(features, label, database_path='street_features.json'):
    try:
        with open(database_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    entry = {'label': label, 'features': features}
    data.append(entry)

    with open(database_path, 'w') as f:
        json.dump(data, f, indent=4)

def build_model(database_path='street_features.json', model_path='street_model.json'):
    with open(database_path, 'r') as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for entry in data:
        grouped[entry['label']].append(entry['features'])

    model = {}
    for label, features_list in grouped.items():
        model_entry = {'mean': {}, 'std': {}}
        for feature in features_list[0].keys():
            values = [sample[feature] for sample in features_list]
            model_entry['mean'][feature] = np.mean(values)
            model_entry['std'][feature] = np.std(values)
        model[label] = model_entry

    with open(model_path, 'w') as f:
        json.dump(model, f, indent=4)
    return model

def predict_street_type(live_features, model_path='street_model.json'):
    with open(model_path, 'r') as f:
        model = json.load(f)

    min_distance = float('inf')
    best_label = None

    for label, stats in model.items():
        distance = 0
        for feature, mean in stats['mean'].items():
            std = stats['std'][feature]
            live_value = live_features.get(feature, 0)
            if std == 0:
                std = 1e-9  # Avoid division by zero
            z = (live_value - mean) / std
            distance += z ** 2
        if distance < min_distance:
            min_distance = distance
            best_label = label

    return best_label

# Example usage
if __name__ == "__main__":
    # Training phase
    files_labels = [
        ('sensor_data_3.txt', 'cobblestone'),
        ('sensor_data_4.txt', 'asphalt')
    ]

    for file_path, label in files_labels:
        features = process_file(file_path)
        if features:
            save_features(features, label)

    # Build model
    build_model()

    # Live prediction example
    live_z_values = [1.2, 1.3, 1.1, 0.9, 0.8, 1.1, 1.2]  # Example live data
    live_features = process_z_values(live_z_values)
    if live_features:
        print("Predicted street type:", predict_street_type(live_features))
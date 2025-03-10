import numpy as np

def compute_similarity(new_data, historical_data):
    """
    Compute similarity between new data and stored historical data.
    :param new_data: Feature vector of the current road segment.
    :param historical_data: List of (feature_vector, road_type) tuples.
    :return: Predicted road type.
    """
    min_distance = float('inf')
    predicted_road = None

    for features, road_type in historical_data:
        distance = np.linalg.norm(np.array(new_data) - np.array(features))  # Euclidean distance
        if distance < min_distance:
            min_distance = distance
            predicted_road = road_type

    return predicted_road

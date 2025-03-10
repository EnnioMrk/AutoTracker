import numpy as np
from scipy.integrate import simpson

def compute_integral_simpson(interval, sampling_rate=250):
    dt = 1 / sampling_rate
    return simpson(interval, dx=dt)


def read_file(file):
    with open(file, "r") as file:
        z_values = []
        for line in file:
            z_values.append(float(line.strip()))

    return z_values

def main(file):
    z_values = read_file(file)

    interval_dict = {}

    zero_crossings = 0
    intervals = []
    current_interval = [z_values[0]]

    std_dev = np.std(z_values, ddof=0)
    mean = np.mean(z_values)
    positive_deviation = sum(a - mean for a in z_values if a > mean)
    negative_deviation = sum(mean - a for a in z_values if a < mean)

    ratio = positive_deviation / negative_deviation if negative_deviation != 0 else float('inf')

    for z in z_values[1:]:
        if (z > 1 and current_interval[0] > 1) or (z < 1 and current_interval[0] < 1):
            current_interval.append(z)
        else:
            intervals.append(current_interval)
            zero_crossings += 1
            current_interval = [z]

    if current_interval:  # Append last interval if it wasn't added
        intervals.append(current_interval)

    for i, interval in enumerate(intervals):
        integral = compute_integral_simpson(interval)
        interval_std_dev = np.std(interval, ddof=0)
        deviation_ratio = (interval_std_dev - std_dev) / std_dev if std_dev != 0 else float('inf')
        interval_dict[i] = {
            "extreme": max(interval) if interval[0] > 1 else min(interval),
            "duration": len(interval),
            "integral": integral,
            "std_dev": deviation_ratio,
            "interval_std_dev": interval_std_dev,
            "mean": np.mean(interval)
        }

    mean_negative_extreme = np.mean([interval_dict[i]["extreme"] for i in interval_dict if interval_dict[i]["extreme"] < 1])
    mean_positive_extreme = np.mean([interval_dict[i]["extreme"] for i in interval_dict if interval_dict[i]["extreme"] > 1])
    mean_integral = np.mean([interval_dict[i]["integral"] for i in interval_dict])

if __name__ == "__main__":
    filename_4 = "no_shoes_6.txt"
    main(filename_4)
    filename_5 = "no_shoes_9.txt"
    main(filename_5)
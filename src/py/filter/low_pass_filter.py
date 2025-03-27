class LowPassFilter:
    def __init__(self, alpha=0.2):
        """
        Initialize the filter with a smoothing factor.
        alpha: Smoothing factor (0 < alpha <= 1). Lower values result in smoother output.
        """
        self.alpha = alpha
        self.filtered_value = None

    def filter(self, new_value):
        """
        Apply the low-pass filter to the new_value.
        If no previous filtered value exists, initialize it with the new_value.
        """
        if self.filtered_value is None:
            self.filtered_value = new_value
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value

class LowPassFilter:
    def __init__(self, alpha=0.2):
        """
        Initialize the filter with a smoothing factor for the basic filter.

        Parameters:
        alpha: Smoothing factor (0 < alpha <= 1). Lower values result in smoother output.
        """
        self.alpha = alpha
        # State for the basic low-pass filter
        self.filtered_basic = None

        # States for the advanced low-pass filter (with coefficients)
        self.filtered_adv = None
        self.prev_input_adv = None

    def filter(self, new_value):
        """
        Apply the basic low-pass filter to the new_value using the fixed alpha.
        If no previous filtered value exists, initialize it with the new_value.

        Parameters:
        new_value: The current input value.

        Returns:
        The filtered output.
        """
        if self.filtered_basic is None:
            self.filtered_basic = new_value
        else:
            self.filtered_basic = self.alpha * new_value + (1 - self.alpha) * self.filtered_basic
        return self.filtered_basic

    def filter_with_coeffs(self, new_value, omega_c, T):
        """
        Apply the alternative first-order low-pass filter that uses both the current and
        previous input values. The difference equation is:

            y[k] = a * y[k-1] + b * (x[k] + x[k-1])

        where:
            a = (2 - T * omega_c) / (2 + T * omega_c)
            b = T * omega_c / (2 + T * omega_c)

        Parameters:
        new_value: The current input value (x[k]).
        omega_c: Cutoff frequency in radians per second.
        T: Sample time in seconds.

        Returns:
        The filtered output y[k].
        """
        # Calculate the filter coefficients based on omega_c and T.
        a = (2 - T * omega_c) / (2 + T * omega_c)
        b = T * omega_c / (2 + T * omega_c)

        # For the first sample, initialize both filtered_adv and prev_input_adv.
        if self.filtered_adv is None:
            self.filtered_adv = new_value
            self.prev_input_adv = new_value
        else:
            self.filtered_adv = a * self.filtered_adv + b * (new_value + self.prev_input_adv)
            self.prev_input_adv = new_value

        return self.filtered_adv

    def reset(self):
        """
        Reset the filter states.
        """
        self.filtered_basic = None
        self.filtered_adv = None
        self.prev_input_adv = None
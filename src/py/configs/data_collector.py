class MainConfig:
    def __init__(self, record=True, visualize=False):
        self.record = record
        self.visualize = visualize
        self.record_data = {
            "calibrated": False
        }
        self.process_data = False
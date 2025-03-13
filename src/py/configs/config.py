SERIAL_PORT = "COM3"
BAUD_RATE = 115200

LABELS = {
    "a": "autobahn",
    "s": "stadt",
    "l": "landstraße",
    "d": "dorf",
}

FIXED_INTERVAL_LENGTH = 50
CALIBRATION_ITERATIONS = 500
H_CALIBRATION_ITERATIONS = 500

# Intervals in seconds
CALIBRATION_INTERVAL = 30
VELOCITY_INTERVAL = 1

# Fine-tuning
VELOCITY_BUFFER = 1.5

ax_bias = -0.04376672
ay_bias = -0.05671166
az_bias = 0.0306987
gx_bias = -0.90627271
gy_bias = -0.32302441
gz_bias = 0.27262613
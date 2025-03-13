#include <FastIMU.h>
#include <Wire.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <ArduinoEigenDense.h>

// Define I2C pins for ESP32
#define SDA_PIN 21
#define SCL_PIN 22

// Select the appropriate IMU from FastIMU
MPU6500 IMU;

// Create calibration data struct
calData calibration;

// Create sensor data struct
AccelData accelData;
GyroData gyroData;
float temp;

const int N = 10;  // Number of samples for averaging
float freqBuffer[N] = {0};  // Buffer to store last N frequency values
int bufferIndex = 0;  // Circular buffer index
static unsigned long lastTime = micros();

void setup() {
  // Initialize Serial
  Serial.begin(115200);
  while (!Serial) {}

  // Initialize I2C
  Wire.begin(SDA_PIN, SCL_PIN);

  // Initialize the MPU-6500
  //Serial.println("Initializing MPU-6500...");

  // Initialize calibration data
  loadCalibration();

  // Initialize sensor with calibration data
  int status = IMU.init(calibration, 0x68);
  if (status < 0) {
    Serial.println("IMU initialization failed!");
    Serial.print("Status: ");
    Serial.println(status);
    while (1) {}
  }
}

void loop() {
  IMU.update();
  IMU.getAccel(&accelData);
  IMU.getGyro(&gyroData);

  Serial.print(accelData.accelX, 10);
  Serial.print("\t");
  Serial.print(accelData.accelY, 10);
  Serial.print("\t");
  Serial.print(accelData.accelZ, 10);
  Serial.print("\t");
  Serial.print(gyroData.gyroX, 10);
  Serial.print("\t");
  Serial.print(gyroData.gyroY, 10);
  Serial.print("\t");
  Serial.println(gyroData.gyroZ, 10);
}

// Function to load calibration data
void loadCalibration() {
  // Set default calibration values
  calibration.accelBias[0] = 0;
  calibration.accelBias[1] = 0;
  calibration.accelBias[2] = 0;

  calibration.gyroBias[0] = 0;
  calibration.gyroBias[1] = 0;
  calibration.gyroBias[2] = 0;
}

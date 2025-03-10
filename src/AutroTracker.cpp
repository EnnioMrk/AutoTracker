#include <Arduino.h>
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

Eigen::Matrix3d R;

float lastValue;
float threshold = 1.1;
int framesBetween = 0;

struct vMeasurement {
  Eigen::Vector3d vec;

  vMeasurement(float a, float b, float c) : vec(a, b, c) {}

  void normalize() {
      double magnitude = vec.norm();
      if (magnitude != 0) {
          vec /= magnitude;
      }
  }
};

std::vector<vMeasurement> measureNAccels(int n) {
  std::vector<vMeasurement> measurements;

  for (int i = 0; i < n; i++) {
      // Read sensor data
      IMU.update();
      IMU.getAccel(&accelData);

      // Store measurement in vector
      measurements.emplace_back(accelData.accelX, accelData.accelY, accelData.accelZ);
  }

  return measurements;
}

Eigen::Vector3d calculateAvgAccel(int iterations = 250) {
  Eigen::Vector3d sumVec = Eigen::Vector3d::Zero();
  std::vector<vMeasurement> measurements = measureNAccels(iterations);

  for (auto& measurement : measurements) {
      measurement.normalize();
      sumVec += measurement.vec;
  }

  Eigen::Vector3d avgVec = sumVec / iterations;
  return avgVec.normalized();  // Ensure the result is a unit vector
}

Eigen::Matrix3d computeRotationMatrix(const Eigen::Vector3d& g_normalized) {
  Eigen::Vector3d v(g_normalized.y(), -g_normalized.x(), 0);
  double theta = std::acos(g_normalized.z());

  double v_norm = v.norm();
  if (v_norm > 1e-6) {
      v /= v_norm;
  } else {
      return Eigen::Matrix3d::Identity();
  }

  Eigen::Matrix3d K;
  K <<  0,      -v.z(),  v.y(),
        v.z(),   0,     -v.x(),
       -v.y(),  v.x(),   0;

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1 - std::cos(theta)) * (K * K);
  return R;
}

void calibrateSensors() {
  Eigen::Vector3d g_normalized = calculateAvgAccel();
  Eigen::Matrix3d R = computeRotationMatrix(g_normalized);
  
  Serial.print("Rotation Matrix R:\n");
  for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
          Serial.print(R(i, j));
          Serial.print(" ");
      }
      Serial.println();
  }
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
  
    Serial.println("Default calibration values set.");
  }
  

void setup() {
  // Initialize Serial
  Serial.begin(115200);
  while (!Serial) {}

  // Initialize I2C
  Wire.begin(SDA_PIN, SCL_PIN);

  // Initialize the MPU-6500
  Serial.println("Initializing MPU-6500...");

  // Initialize calibration data
  loadCalibration();

  // Initialize sensor with calibration data
  int status = IMU.init(calibration);
  if (status < 0) {
    Serial.println("IMU initialization failed!");
    Serial.print("Status: ");
    Serial.println(status);
    while (1) {}
  }

  // Print success message
  Serial.println("MPU-6500 initialization successful!");

  // Compute and store the rotation matrix
  Eigen::Vector3d g_normalized = calculateAvgAccel();
  R = computeRotationMatrix(g_normalized);

  // Print rotation matrix for debugging
  Serial.println("Rotation Matrix R:");
  for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
          Serial.print(R(i, j), 6);
          Serial.print(" ");
      }
      Serial.println();
  }


}

void loop() {
  // Read sensor data
  IMU.update();

  // Get data without returning success/failure flags
  IMU.getAccel(&accelData);
  IMU.getGyro(&gyroData);

  // Get temperature
  //temp = IMU.getTemp();

  // Convert raw sensor data to Eigen vectors
  Eigen::Vector3d accelVec(accelData.accelX, accelData.accelY, accelData.accelZ);
  Eigen::Vector3d gyroVec(gyroData.gyroX, gyroData.gyroY, gyroData.gyroZ);

  Eigen::Vector3d accelTransformed = R * accelVec;
  Eigen::Vector3d gyroTransformed = R * gyroVec;

  // Print formatted data for Serial Plotter
  //double gyroAvg = (gyroData.gyroX + gyroData.gyroY + gyroData.gyroZ) / 3;
  //double accelAvg = (accelData.accelX + accelData.accelY + accelData.accelZ) / 3;
  /*Serial.print("\t");
  Serial.print(gyroAvg);
  Serial.print("\t");
  Serial.print(accelAvg);
  Serial.print("\t");*/
  Serial.print(accelTransformed.x(), 4);
  Serial.print("\t");
  Serial.print(accelTransformed.y(), 4);
  Serial.print("\t");
  Serial.println(accelTransformed.z());
  //Serial.print("\t");
  //Serial.println(accelTransformed.z(), 4);
  /*Serial.print("\t");
  Serial.print(gyroData.gyroX, 4);
  Serial.print("\t");
  Serial.print(gyroData.gyroY, 4);
  Serial.print("\t");
  Serial.print(gyroData.gyroZ, 4);
  Serial.print("\t");
  Serial.println(temp, 4);*/

  if(lastValue>threshold) {
    if(accelTransformed.z()>threshold) {
      framesBetween++;
    } else {
      if(framesBetween>=10) {
        //Serial.println(framesBetween);
      }
      framesBetween = 0;
    }
  }
  lastValue = accelTransformed.z();
}


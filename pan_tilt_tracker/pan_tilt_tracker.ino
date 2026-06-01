#include <ESP32Servo.h>
#include <Wire.h>
#include <Adafruit_VL53L0X.h>

// =====================================
// PIN CONFIG
// =====================================

#define PAN_CAM_PIN 33
#define TILT_CAM_PIN 13

#define PAN_LAUNCH_PIN 15
#define TILT_LAUNCH_PIN 16

#define MOTION_PIN 27
#define BUZZER_PIN 23

// =====================================
// SERVOS
// =====================================

Servo panCamServo;
Servo tiltCamServo;

Servo panLaunchServo;
Servo tiltLaunchServo;

int panAngle = 90;
int tiltAngle = 90;

// =====================================
// VL53L0X
// =====================================

Adafruit_VL53L0X lox;

// =====================================
// TIMER
// =====================================

unsigned long lastSend = 0;
const int SEND_INTERVAL = 100;

// =====================================

void setup()
{
  Serial.begin(115200);

  Wire.begin(21, 22);

  pinMode(MOTION_PIN, INPUT);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // ==========================
  // Camera Pan Tilt
  // ==========================

  panCamServo.setPeriodHertz(50);
  tiltCamServo.setPeriodHertz(50);

  panCamServo.attach(
      PAN_CAM_PIN,
      500,
      2500);

  tiltCamServo.attach(
      TILT_CAM_PIN,
      500,
      2500);

  // ==========================
  // Launcher Pan Tilt
  // ==========================

  panLaunchServo.setPeriodHertz(50);
  tiltLaunchServo.setPeriodHertz(50);

  panLaunchServo.attach(
      PAN_LAUNCH_PIN,
      500,
      2500);

  tiltLaunchServo.attach(
      TILT_LAUNCH_PIN,
      500,
      2500);

  // ==========================
  // Center Position
  // ==========================

  panCamServo.write(90);
  tiltCamServo.write(90);

  panLaunchServo.write(90);
  tiltLaunchServo.write(90);

  // ==========================
  // VL53L0X
  // ==========================

  if (!lox.begin())
  {
    Serial.println("VL53L0X_ERROR");
  }
  else
  {
    Serial.println("VL53L0X_OK");
  }

  delay(1000);
}

// =====================================

void readSerialCommand()
{
  if (!Serial.available())
    return;

  String cmd =
      Serial.readStringUntil('\n');

  int pIndex =
      cmd.indexOf("P:");

  int tIndex =
      cmd.indexOf(",T:");

  if (pIndex == -1 || tIndex == -1)
    return;

  int newPan =
      cmd.substring(
             pIndex + 2,
             tIndex)
          .toInt();

  int newTilt =
      cmd.substring(
             tIndex + 3)
          .toInt();

  // ==================================
  // BÁO CÒI NẾU PYTHON GỬI NGOÀI 0-180
  // ==================================

  if (newPan < 0 || newPan > 180)
  {
    digitalWrite(BUZZER_PIN, HIGH);
  }
  else
  {
    digitalWrite(BUZZER_PIN, LOW);
  }

  // ==========================
  // Full Range
  // ==========================

  newPan =
      constrain(
          newPan,
          0,
          180);

  newTilt =
      constrain(
          newTilt,
          0,
          180);

  panAngle = newPan;
  tiltAngle = newTilt;

  Serial.print("PAN=");
  Serial.print(panAngle);

  Serial.print(" TILT=");
  Serial.println(tiltAngle);

  // ==========================
  // CAMERA
  // ==========================

  panCamServo.write(
      panAngle);

  tiltCamServo.write(
      tiltAngle);

  // ==========================
  // LAUNCHER
  // ==========================

  panLaunchServo.write(
      panAngle);

  tiltLaunchServo.write(
      tiltAngle);
}

// =====================================

void sendSensorData()
{
  if (
      millis() - lastSend < SEND_INTERVAL)
    return;

  lastSend = millis();

  VL53L0X_RangingMeasurementData_t measure;

  lox.rangingTest(
      &measure,
      false);

  uint16_t distance = 8190;

  if (
      measure.RangeStatus != 4)
  {
    distance =
        measure.RangeMilliMeter;
  }

  int motion =
      digitalRead(
          MOTION_PIN);

  Serial.print("D:");
  Serial.print(distance);

  Serial.print(",M:");
  Serial.println(motion);
}

// =====================================

void loop()
{
  readSerialCommand();

  sendSensorData();
}
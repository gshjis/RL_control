#ifndef AS5600_H
#define AS5600_H

#include <Arduino.h>
#include <Wire.h>
#include "config.h"

class AS5600 {
public:
  AS5600(uint8_t address = AS5600_ADDR_PENDULUM);

  bool begin();
  bool readRawAngle(uint16_t& raw_angle);
  float readAngleRad();

  // Утилиты
  static float rawToRad(uint16_t raw);
  static int16_t angleDiff(int16_t current, int16_t previous);

private:
  uint8_t _address;
  uint16_t _last_raw = 0;
  bool _initialized = false;

  // Регистры AS5600
  static constexpr uint8_t REG_RAW_ANGLE_H = 0x0C;
  static constexpr uint8_t REG_RAW_ANGLE_L = 0x0D;
};

#endif // AS5600_H
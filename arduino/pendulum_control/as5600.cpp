#include "as5600.h"

AS5600::AS5600(uint8_t address) : _address(address) {}

bool AS5600::begin() {
  Wire.begin();
  Wire.setClock(I2C_FREQ_HZ);
  
  // Проверка связи: пробуем прочитать угол
  uint16_t dummy;
  _initialized = readRawAngle(dummy);
  return _initialized;
}

bool AS5600::readRawAngle(uint16_t& raw_angle) {
  Wire.beginTransmission(_address);
  Wire.write(REG_RAW_ANGLE_H);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }
  
  if (Wire.requestFrom(_address, (uint8_t)2) != 2) {
    return false;
  }
  
  uint8_t high = Wire.read();
  uint8_t low = Wire.read();
  
  raw_angle = (high << 8) | low;
  _last_raw = raw_angle;
  return true;
}

float AS5600::readAngleRad() {
  uint16_t raw;
  if (!readRawAngle(raw)) {
    return NAN;
  }
  return rawToRad(raw);
}

float AS5600::rawToRad(uint16_t raw) {
  return raw * AS5600_ANGLE_TO_RAD;
}

int16_t AS5600::angleDiff(int16_t current, int16_t previous) {
  int16_t diff = current - previous;
  // Обработка перехода через 0/4095
  if (diff > 2048) diff -= 4096;
  if (diff < -2048) diff += 4096;
  return diff;
}
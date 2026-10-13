#include "tb6612.h"
#include "config.h"

TB6612::TB6612() {}

void TB6612::begin() {
  pinMode(PIN_MOTOR_PWM, OUTPUT);
  pinMode(PIN_MOTOR_DIR, OUTPUT);
  pinMode(PIN_MOTOR_STBY, OUTPUT);
  
  // Timer1 настройка для PWM на pin 9 (OC1A) — 500 Гц не подходит для PWM,
  // используем стандартный 490 Гц или настраиваем Timer1 в Fast PWM mode
  // Здесь используем стандартный analogWrite (490 Гц на pin 9)
  
  digitalWrite(PIN_MOTOR_STBY, LOW);  // Отключен по умолчанию
  digitalWrite(PIN_MOTOR_DIR, LOW);
  analogWrite(PIN_MOTOR_PWM, 0);
}

void TB6612::setForce(float force_n) {
  _current_force = constrain(force_n, -MAX_FORCE_N, MAX_FORCE_N);
  
  if (!_enabled) return;
  
  // Направление
  if (_current_force >= 0) {
    digitalWrite(PIN_MOTOR_DIR, HIGH);
  } else {
    digitalWrite(PIN_MOTOR_DIR, LOW);
  }
  
  // PWM (0-255)
  uint8_t pwm = forceToPWM(fabsf(_current_force));
  analogWrite(PIN_MOTOR_PWM, pwm);
}

uint8_t TB6612::forceToPWM(float force_n) const {
  // Линейное отображение: 0..MAX_FORCE_N → 0..255
  // Можно добавить deadband compensation
  float ratio = force_n / MAX_FORCE_N;
  uint8_t pwm = (uint8_t)roundf(ratio * 255.0f);
  return constrain(pwm, 0, 255);
}

void TB6612::enable() {
  _enabled = true;
  digitalWrite(PIN_MOTOR_STBY, HIGH);
}

void TB6612::disable() {
  _enabled = false;
  digitalWrite(PIN_MOTOR_STBY, LOW);
  analogWrite(PIN_MOTOR_PWM, 0);
}

void TB6612::brake() {
  // Короткое замыкание обмоток (LOW на обоих входах + STBY HIGH)
  digitalWrite(PIN_MOTOR_DIR, LOW);
  analogWrite(PIN_MOTOR_PWM, 255);  // 100% PWM = brake
}
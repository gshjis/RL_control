/*
 * Pendulum Control — Arduino Uno (ATmega328P)
 * 
 * Квантованная PPO политика для управления инвертированным маятником.
 * 
 * Железо:
 *   - Arduino Uno (ATmega328P, 16 MHz)
 *   - AS5600 x2 (I2C): маятник θ1 + тележка x
 *   - TB6612FNG: DC мотор (PWM + DIR)
 * 
 * Частота: 100 Гц (Timer1 CTC interrupt), как controller.dt при обучении
 * НС: MLP произвольной архитектуры (INT8), задаётся model_weights.h
 */

#include <Wire.h>
#include <avr/wdt.h>
#include <avr/interrupt.h>

#include "config.h"
#include "as5600.h"
#include "sensor_block.h"
#include "inference.h"
#include "tb6612.h"
#include "model_weights.h"  // Генерируется export_quantize.py

// ============================================================================
// ГЛОБАЛЬНЫЕ ОБЪЕКТЫ
// ============================================================================

AS5600 encoder_pendulum(AS5600_ADDR_PENDULUM);
AS5600 encoder_cart(AS5600_ADDR_CART);  // Нужен мультиплексор!

SensorBlock sensor_block(1.0f / CONTROL_FREQ_HZ);
QuantizedInference inference(&g_quantized_model);
TB6612 motor;

// Телеметрия
Telemetry telemetry;

// Флаги
volatile bool _control_flag = false;
volatile uint32_t _loop_count = 0;
bool _emergency_stop = false;

// ============================================================================
// TIMER1 CTC INTERRUPT — 100 Гц
// ============================================================================

void setupTimer1() {
  noInterrupts();
  
  // Timer1: CTC mode, prescaler 64
  // 16 MHz / 64 = 250 kHz
  // OCR1A = 250000 / 100 - 1 = 2499
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;
  OCR1A = 2499;  // 100 Гц при 16 МГц / 64
  
  TCCR1B |= (1 << WGM12);        // CTC mode
  TCCR1B |= (1 << CS11) | (1 << CS10);  // Prescaler 64
  TIMSK1 |= (1 << OCIE1A);       // Enable compare match interrupt
  
  interrupts();
}

ISR(TIMER1_COMPA_vect) {
  _control_flag = true;
  _loop_count++;
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  // Отключаем watchdog на время инициализации
  wdt_disable();
  
  // Serial для отладки
#ifdef DEBUG_SERIAL
  Serial.begin(SERIAL_BAUD);
  while (!Serial) {}
  DEBUG_PRINTLN(F("Pendulum Control starting..."));
#endif
  
  // I2C
  Wire.begin();
  Wire.setClock(I2C_FREQ_HZ);
  
  // Энкодеры
  if (!encoder_pendulum.begin()) {
#ifdef DEBUG_SERIAL
    DEBUG_PRINTLN(F("ERROR: Pendulum encoder not found!"));
#endif
    _emergency_stop = true;
  }
  
  // Энкодер тележки имеет отдельный адрес (AS5600_ADDR_CART)
  if (!encoder_cart.begin()) {
#ifdef DEBUG_SERIAL
    DEBUG_PRINTLN(F("ERROR: Cart encoder not found!"));
#endif
    _emergency_stop = true;
  }

#ifdef DEBUG_SERIAL
  DEBUG_PRINTLN(F("Encoders initialized"));
#endif
  
  // Sensor block
  sensor_block.reset();
  sensor_block.setSeed(micros());
  
  // Мотор
  motor.begin();
  
  // Timer1 для 100 Гц
  setupTimer1();
  
  // Watchdog (8 сек)
  wdt_enable(WDTO_8S);
  
#ifdef DEBUG_SERIAL
  DEBUG_PRINTLN(F("Setup complete. Starting control loop."));
#endif
  
  // Включаем мотор
  motor.enable();
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  // Сбрасываем watchdog
  wdt_reset();
  
  // Ждём флаг от таймера (100 Гц)
  if (!_control_flag) return;
  _control_flag = false;
  
  // Аварийная остановка
  if (_emergency_stop) {
    motor.disable();
    return;
  }
  
  // --- 1. Чтение энкодеров ---
  uint16_t raw_pendulum, raw_cart;
  bool ok_pendulum = encoder_pendulum.readRawAngle(raw_pendulum);
  bool ok_cart = encoder_cart.readRawAngle(raw_cart);
  
  if (!ok_pendulum || !ok_cart) {
    // Потеря связи с энкодером
    motor.disable();
    _emergency_stop = true;
#ifdef DEBUG_SERIAL
    DEBUG_PRINTLN(F("EMERGENCY: Encoder communication lost!"));
#endif
    return;
  }
  
  // --- 2. Sensor Block (квантование/шум/дифф/ФНЧ) ---
  sensor_block.update(raw_pendulum, raw_cart, telemetry);
  
  // --- 3. Построение признаков для НС ---
  // Порядок как при обучении: [sinθ1, cosθ1, dθ1, dx], в единицах обучения.
  // Нормализация (статистика VecNormalize) и квантование — внутри forward().
#if NN_INPUT_DIM != 4
#error "Скетч строит 4 признака; для другой размерности обновите этот блок"
#endif
  float features[NN_INPUT_DIM];
  features[0] = telemetry.sin_t1;
  features[1] = telemetry.cos_t1;
  features[2] = telemetry.dtheta1;
  features[3] = telemetry.dx;

  // --- 4. INT8 Inference ---
  float force = inference.forward(features);
  
  // --- 5. Безопасность ---
  // Лимит силы
  force = constrain(force, -MAX_FORCE_N, MAX_FORCE_N);
  
  // --- 6. Применяем к мотору ---
  motor.setForce(force);
  
  // --- 7. Отладка ---
#ifdef DEBUG_SERIAL
  if (_loop_count % 100 == 0) {  // 1 раз в секунду
    Serial.print(F("x=")); Serial.print(telemetry.x, 4);
    Serial.print(F(" sin=")); Serial.print(telemetry.sin_t1, 3);
    Serial.print(F(" cos=")); Serial.print(telemetry.cos_t1, 3);
    Serial.print(F(" dθ=")); Serial.print(telemetry.dtheta1, 3);
    Serial.print(F(" dx=")); Serial.print(telemetry.dx, 3);
    Serial.print(F(" force=")); Serial.println(force, 3);
  }
#endif
}
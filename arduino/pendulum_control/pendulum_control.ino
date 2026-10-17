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
 * Частота: 500 Гц (Timer1 CTC interrupt)
 * НС: MLP 4→32→1 (INT8, Tanh)
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
SensorBlock::Telemetry telemetry;

// Флаги
volatile bool _control_flag = false;
volatile uint32_t _loop_count = 0;
bool _emergency_stop = false;

// ============================================================================
// TIMER1 CTC INTERRUPT — 500 Гц
// ============================================================================

void setupTimer1() {
  noInterrupts();
  
  // Timer1: CTC mode, prescaler 64
  // 16 MHz / 64 = 250 kHz
  // OCR1A = 250000 / 500 - 1 = 499
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;
  OCR1A = 499;  // 500 Гц при 16 МГц / 64
  
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
  
  // Энкодер тележки — тот же адрес, нужен мультиплексор!
  // Пока используем заглушку
#ifdef DEBUG_SERIAL
  DEBUG_PRINTLN(F("Encoders initialized"));
#endif
  
  // Sensor block
  sensor_block.reset();
  sensor_block.setSeed(micros());
  
  // Мотор
  motor.begin();
  
  // Timer1 для 500 Гц
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
  
  // Ждём флаг от таймера (500 Гц)
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
  
  // Для тележки — пока заглушка (нужен мультиплексор I2C)
  raw_cart = 0;  // TODO: TCA9548A или SoftI2C
  bool ok_cart = true;
  
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
  // [sinθ1, cosθ1, dθ1, dx] — в Q7.8 (int16_t)
  int16_t features_q[NN_INPUT_DIM];
  
  // sinθ1, cosθ1 уже в [-1, 1] → Q7.8
  features_q[0] = (int16_t)roundf(telemetry.sin_t1 * 256.0f);
  features_q[1] = (int16_t)roundf(telemetry.cos_t1 * 256.0f);
  
  // dθ1 в рад/с — масштабируем (предполагаем ±20 рад/с)
  features_q[2] = (int16_t)roundf(telemetry.dtheta1 * 12.8f);  // 256/20 ≈ 12.8
  
  // dx в м/с — масштабируем (предполагаем ±2 м/с)
  features_q[3] = (int16_t)roundf(telemetry.dx * 128.0f);  // 256/2 = 128
  
  // Клиппинг в int16_t
  for (int i = 0; i < NN_INPUT_DIM; i++) {
    features_q[i] = constrain(features_q[i], -32768, 32767);
  }
  
  // --- 4. INT8 Inference ---
  // Конвертируем Q7.8 → int8_t для входа НС
  int8_t input_q8[NN_INPUT_DIM];
  for (int i = 0; i < NN_INPUT_DIM; i++) {
    // Масштабируем из Q7.8 в диапазон int8_t (-128..127)
    // input_scale из модели учитывает это
    input_q8[i] = (int8_t)constrain(features_q[i] >> 3, -128, 127);  // /8 = /256*32
  }
  
  float force = inference.forwardQuantized(input_q8);
  
  // --- 5. Безопасность ---
  // Лимит позиции тележки
  if (fabsf(telemetry.x) > CART_POS_LIMIT) {
    force = 0;
    motor.disable();
    _emergency_stop = true;
#ifdef DEBUG_SERIAL
    DEBUG_PRINTLN(F("EMERGENCY: Cart position limit!"));
#endif
  }
  
  // Лимит силы
  force = constrain(force, -MAX_FORCE_N, MAX_FORCE_N);
  
  // --- 6. Применяем к мотору ---
  motor.setForce(force);
  
  // --- 7. Отладка ---
#ifdef DEBUG_SERIAL
  if (_loop_count % 500 == 0) {  // 1 раз в секунду
    DEBUG_PRINT(F("θ1=")); DEBUG_PRINT(telemetry.x, 4);
    DEBUG_PRINT(F(" sin=")); DEBUG_PRINT(telemetry.sin_t1, 3);
    DEBUG_PRINT(F(" cos=")); DEBUG_PRINT(telemetry.cos_t1, 3);
    DEBUG_PRINT(F(" dθ=")); DEBUG_PRINT(telemetry.dtheta1, 3);
    DEBUG_PRINT(F(" dx=")); DEBUG_PRINT(telemetry.dx, 3);
    DEBUG_PRINT(F(" force=")); DEBUG_PRINTLN(force, 3);
  }
#endif
}
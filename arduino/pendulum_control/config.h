#ifndef CONFIG_H
#define CONFIG_H

#pragma once

// ============================================================================
// ПИНЫ
// ============================================================================

// TB6612FNG Motor Driver
#define PIN_MOTOR_PWM       9   // OC1A (Timer1) — PWM выход
#define PIN_MOTOR_DIR       2   // DIR вход
#define PIN_MOTOR_STBY      3   // STBY (standby) — HIGH = включен

// AS5600 I2C (на шине Wire: A4=SDA, A5=SCL)
#define AS5600_ADDR_PENDULUM  0x36  // Адрес энкодера маятника (θ1)
#define AS5600_ADDR_CART      0x36  // Адрес энкодера тележки (x) — нужен мультиплексор!

// ============================================================================
// ЧАСТОТЫ И ТАЙМИНГИ
// ============================================================================

#define CONTROL_FREQ_HZ     500       // Частота цикла управления (Гц)
#define CONTROL_PERIOD_US   (1000000L / CONTROL_FREQ_HZ)  // 2000 мкс

#define I2C_FREQ_HZ         400000L   // 400 кГц I2C (Fast Mode+)
#define SERIAL_BAUD         115200    // Для отладки

// ============================================================================
// ФИЗИЧЕСКИЕ КОНСТАНТЫ (совпадают с обучением)
// ============================================================================

// Максимальная сила (Н) — должна совпадать с ControllerConfig.max_force
#define MAX_FORCE_N         24.0f

// Параметры AS5600
#define AS5600_CPR          4096      // Counts per revolution (12 бит)
#define AS5600_ANGLE_TO_RAD (2.0f * PI / AS5600_CPR)

// ============================================================================
// SENSOR BLOCK (эквивалент C++ SensorBlock)
// ============================================================================

// Квантование энкодера (эмуляция C++ SensorBlock)
#define ENCODER_QUANTIZE    1         // 1 = включать квантование

// Шум позиции (std, рад) — из SensorConfig.noise_std_q
#define NOISE_STD_Q_THETA1  0.002f
#define NOISE_STD_Q_CART    0.0005f

// Шум скорости (std, рад/с) — из SensorConfig.noise_std_dq
#define NOISE_STD_DQ_THETA1 0.01f
#define NOISE_STD_DQ_CART   0.005f

// Дифференцирующий фильтр (LPF на производной) — cutoff Hz
#define DIFF_CUTOFF_HZ      40.0f

// ФНЧ на выходе телеметрии — cutoff Hz
#define FILTER_CUTOFF_HZ    50.0f

// ============================================================================
// НЕЙРОСЕТЬ
// ============================================================================
// Размеры и архитектура приходят из сгенерированного model_weights.h
// (NN_INPUT_DIM, NN_NUM_LAYERS, NN_MAX_DIM) — правьте их экспортёром,
// а не здесь. Формат активаций: Q7.8 (int16, значение = int / 256).

// ============================================================================
// БЕЗОПАСНОСТЬ
// ============================================================================

// Лимит позиции тележки (м) — аварийная остановка
#define CART_POS_LIMIT      0.5f

// Watchdog timeout (мс) — 0 = отключен
#define WATCHDOG_TIMEOUT_MS 8000

// ============================================================================
// ОТЛАДКА
// ============================================================================

// #define DEBUG_SERIAL 1  // Раскомментировать для Serial вывода

#ifdef DEBUG_SERIAL
  #define DEBUG_PRINT(x)   Serial.print(x)
  #define DEBUG_PRINTLN(x) Serial.println(x)
#else
  #define DEBUG_PRINT(x)
  #define DEBUG_PRINTLN(x)
#endif

#endif // CONFIG_H
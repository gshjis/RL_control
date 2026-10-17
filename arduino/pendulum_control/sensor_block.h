#ifndef SENSOR_BLOCK_H
#define SENSOR_BLOCK_H

#include <Arduino.h>
#include "config.h"

// ============================================================================
// SensorBlock — перенос C++ SensorBlock на Arduino
// Пайплайн: raw_angle → квантование → +шум → дифференцирование (LPF) → +шум → ФНЧ
// ============================================================================

struct Telemetry {
  float x;          // Позиция тележки (м)
  float cos_t1;     // cos(θ1)
  float sin_t1;     // sin(θ1)
  float cos_t2;     // cos(θ2) — не используется в single
  float sin_t2;     // sin(θ2) — не используется в single
  float dx;         // Скорость тележки (м/с)
  float dtheta1;    // Угловая скорость θ1 (рад/с)
  float dtheta2;    // Угловая скорость θ2 — не используется
};

class SensorBlock {
public:
  SensorBlock(float dt = 0.002f);  // dt = 1/500 = 0.002 с

  void reset();
  void setSeed(uint32_t seed);
  
  // Основной метод: обновляет телеметрию из сырых энкодеров
  // Возвращает true если данные валидны
  bool update(uint16_t raw_pendulum, uint16_t raw_cart, Telemetry& out);

  // Геттеры для отладки
  float getTheta1Rad() const { return _theta1_filtered; }
  float getDTheta1() const { return _dtheta1_filtered; }
  float getCartPos() const { return _x_filtered; }
  float getCartVel() const { return _dx_filtered; }

private:
  // Внутреннее состояние
  float _dt;
  
  // Сырые значения
  uint16_t _raw_pendulum_prev = 0;
  uint16_t _raw_cart_prev = 0;
  bool _first_sample = true;

  // После квантования + шума позиции
  float _theta1_quantized = 0;
  float _x_quantized = 0;

  // Дифференцирующий фильтр (LPF на производной)
  float _dtheta1_raw = 0;
  float _dtheta1_lpf = 0;
  float _dx_raw = 0;
  float _dx_lpf = 0;

  // ФНЧ на выходе
  float _theta1_filtered = 0;
  float _dtheta1_filtered = 0;
  float _x_filtered = 0;
  float _dx_filtered = 0;

  // RNG для шума
  uint32_t _rng_state = 1;
  
  // Коэффициенты фильтров (вычисляются в конструкторе)
  float _alpha_diff;   // Для дифференцирующего LPF
  float _alpha_filter; // Для выходного ФНЧ

  // Вспомогательные
  float randn(float stddev);
  float quantizeAngle(float angle_rad, float resolution_rad);
  float quantizeCart(float pos_m, float resolution_m);
};

#endif // SENSOR_BLOCK_H
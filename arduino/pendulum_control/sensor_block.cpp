#include "sensor_block.h"
#include "config.h"
#include <math.h>

SensorBlock::SensorBlock(float dt) : _dt(dt) {
  // Коэффициенты экспоненциального сглаживания: alpha = dt / (tau + dt)
  // tau = 1 / (2 * PI * fc)
  float tau_diff = 1.0f / (2.0f * PI * DIFF_CUTOFF_HZ);
  _alpha_diff = _dt / (tau_diff + _dt);

  float tau_filter = 1.0f / (2.0f * PI * FILTER_CUTOFF_HZ);
  _alpha_filter = _dt / (tau_filter + _dt);
}

void SensorBlock::reset() {
  _raw_pendulum_prev = 0;
  _raw_cart_prev = 0;
  _first_sample = true;
  _theta1_quantized = 0;
  _x_quantized = 0;
  _dtheta1_raw = 0;
  _dtheta1_lpf = 0;
  _dx_raw = 0;
  _dx_lpf = 0;
  _theta1_filtered = 0;
  _dtheta1_filtered = 0;
  _x_filtered = 0;
  _dx_filtered = 0;
  _rng_state = 1;
}

void SensorBlock::setSeed(uint32_t seed) {
  _rng_state = seed ? seed : 1;
}

// Простой xorshift32 RNG
float SensorBlock::randn(float stddev) {
  // Box-Muller transform для нормального распределения
  static float spare = 0;
  static bool has_spare = false;
  
  if (has_spare) {
    has_spare = false;
    return spare * stddev;
  }
  
  // xorshift32
  _rng_state ^= _rng_state << 13;
  _rng_state ^= _rng_state >> 17;
  _rng_state ^= _rng_state << 5;
  
  float u1 = (_rng_state & 0xFFFFFF) / 16777216.0f;
  _rng_state ^= _rng_state << 13;
  _rng_state ^= _rng_state >> 17;
  _rng_state ^= _rng_state << 5;
  float u2 = (_rng_state & 0xFFFFFF) / 16777216.0f;
  
  float mag = sqrtf(-2.0f * logf(u1 + 1e-10f));
  float z0 = mag * cosf(2.0f * PI * u2);
  float z1 = mag * sinf(2.0f * PI * u2);
  
  spare = z1;
  has_spare = true;
  
  return z0 * stddev;
}

float SensorBlock::quantizeAngle(float angle_rad, float resolution_rad) {
  if (resolution_rad <= 0) return angle_rad;
  return roundf(angle_rad / resolution_rad) * resolution_rad;
}

float SensorBlock::quantizeCart(float pos_m, float resolution_m) {
  if (resolution_m <= 0) return pos_m;
  return roundf(pos_m / resolution_m) * resolution_m;
}

bool SensorBlock::update(uint16_t raw_pendulum, uint16_t raw_cart, Telemetry& out) {
  // --- 1. Сырые углы в радианы ---
  float theta1_raw = AS5600::rawToRad(raw_pendulum);  // 0..2π
  float x_raw = AS5600::rawToRad(raw_cart) * 0.01f;   // Пример: 1 оборот = 1 см (настроить!)
  
  // Приводим θ1 к [-π, π] для удобства (cos/sin работают с любым)
  // Но для квантования используем raw 0..2π
  
  // --- 2. Квантование + шум позиции ---
  float theta1_resolution = 2.0f * PI / AS5600_CPR;  // ~0.0015 рад
  float cart_resolution = 0.0001f;  // 0.1 мм
  
  float theta1_quant = quantizeAngle(theta1_raw, theta1_resolution);
  float x_quant = quantizeCart(x_raw, cart_resolution);
  
  // Добавляем шум позиции
#if ENCODER_QUANTIZE
  _theta1_quantized = theta1_quant + randn(NOISE_STD_Q_THETA1);
  _x_quantized = x_quant + randn(NOISE_STD_Q_CART);
#else
  _theta1_quantized = theta1_raw + randn(NOISE_STD_Q_THETA1);
  _x_quantized = x_raw + randn(NOISE_STD_Q_CART);
#endif
  
  // --- 3. Дифференцирование (конечные разности + LPF) ---
  if (_first_sample) {
    _raw_pendulum_prev = raw_pendulum;
    _raw_cart_prev = raw_cart;
    _first_sample = false;
    _dtheta1_raw = 0;
    _dx_raw = 0;
  } else {
    // Разность углов с учётом перехода через 0
    int16_t diff_pendulum = AS5600::angleDiff(raw_pendulum, _raw_pendulum_prev);
    int16_t diff_cart = AS5600::angleDiff(raw_cart, _raw_cart_prev);
    
    _dtheta1_raw = diff_pendulum * AS5600_ANGLE_TO_RAD / _dt;
    _dx_raw = diff_cart * AS5600_ANGLE_TO_RAD * 0.01f / _dt;  // 0.01f = м/оборот
    
    _raw_pendulum_prev = raw_pendulum;
    _raw_cart_prev = raw_cart;
  }
  
  // LPF на производной (дифференцирующий фильтр)
  _dtheta1_lpf = _alpha_diff * _dtheta1_raw + (1.0f - _alpha_diff) * _dtheta1_lpf;
  _dx_lpf = _alpha_diff * _dx_raw + (1.0f - _alpha_diff) * _dx_lpf;
  
  // Добавляем шум скорости
  float dtheta1_noisy = _dtheta1_lpf + randn(NOISE_STD_DQ_THETA1);
  float dx_noisy = _dx_lpf + randn(NOISE_STD_DQ_CART);
  
  // --- 4. ФНЧ на выходе ---
  _theta1_filtered = _alpha_filter * _theta1_quantized + (1.0f - _alpha_filter) * _theta1_filtered;
  _dtheta1_filtered = _alpha_filter * dtheta1_noisy + (1.0f - _alpha_filter) * _dtheta1_filtered;
  _x_filtered = _alpha_filter * _x_quantized + (1.0f - _alpha_filter) * _x_filtered;
  _dx_filtered = _alpha_filter * dx_noisy + (1.0f - _alpha_filter) * _dx_filtered;
  
  // --- 5. Формируем выходную телеметрию ---
  out.x = _x_filtered;
  out.cos_t1 = cosf(_theta1_filtered);
  out.sin_t1 = sinf(_theta1_filtered);
  out.cos_t2 = 1.0f;  // single pendulum: θ2 = 0
  out.sin_t2 = 0.0f;
  out.dx = _dx_filtered;
  out.dtheta1 = _dtheta1_filtered;
  out.dtheta2 = 0.0f;
  
  return true;
}
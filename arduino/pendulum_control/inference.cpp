#include "inference.h"
#include "config.h"
#include <math.h>

// Tanh lookup table для Q7.8 (-128..127 -> -128..127)
// Генерируется один раз, здесь упрощённая версия
int8_t QuantizedInference::tanh_q78(int16_t x_q78) {
  // Ограничиваем вход
  if (x_q78 >= 32767) return 127;   // tanh(127.99) ≈ 1.0
  if (x_q78 <= -32768) return -128; // tanh(-128.0) ≈ -1.0
  
  // Преобразуем в float, считаем tanh, обратно в Q7.8
  float x = x_q78 / 256.0f;
  float y = tanhf(x);
  return (int8_t)roundf(y * 256.0f);
}

QuantizedInference::QuantizedInference(const QuantizedModel* model) : _model(model) {}

float QuantizedInference::forward(const float input[NN_INPUT_DIM]) {
  // Квантуем вход: float → int8_t
  int8_t input_q[NN_INPUT_DIM];
  for (int i = 0; i < NN_INPUT_DIM; i++) {
    float scaled = input[i] * _model->input_scale;
    int16_t q = (int16_t)roundf(scaled) + _model->input_zp;
    // Клиппинг в int8_t
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    input_q[i] = (int8_t)q;
  }
  
  return forwardQuantized(input_q);
}

float QuantizedInference::forwardQuantized(const int8_t input[NN_INPUT_DIM]) {
  const QuantizedModel* m = _model;
  
  // === Layer 1: Input(4) → Hidden(32) ===
  for (int j = 0; j < NN_HIDDEN_DIM; j++) {
    // MAC: sum_i(w1[j][i] * input[i]) + b1[j]
    int16_t acc = 0;
    for (int i = 0; i < NN_INPUT_DIM; i++) {
      int8_t w = pgm_read_byte(&m->w1[j * NN_INPUT_DIM + i]);
      int8_t x = input[i];
      acc += (int16_t)w * (int16_t)x;
    }
    
    // Добавляем bias (уже в int8_t, масштабируем)
    int8_t b = pgm_read_byte(&m->b1[j]);
    acc += (int16_t)b * 256;  // bias в том же масштабе что веса*вход
    
    // Масштабирование: (w_scale * in_scale) / hidden_scale
    float scale = (m->w1_scale * m->input_scale) / (m->w1_scale * m->input_scale); // = 1.0 для простоты
    // В правильной реализации нужен per-tensor scale
    
    // Применяем scale и zero-point
    float acc_f = acc * m->w1_scale * m->input_scale;
    int16_t acc_q = (int16_t)roundf(acc_f * 256.0f) + m->hidden_zp * 256;
    
    // Клиппинг
    if (acc_q > 32767) acc_q = 32767;
    if (acc_q < -32768) acc_q = -32768;
    
    _hidden_acc[j] = acc_q;
    _hidden_out[j] = tanh_q78(acc_q);
  }
  
  // === Layer 2: Hidden(32) → Output(1) ===
  int16_t acc = 0;
  for (int j = 0; j < NN_HIDDEN_DIM; j++) {
    int8_t w = pgm_read_byte(&m->w2[j]);
    int8_t h = _hidden_out[j];
    acc += (int16_t)w * (int16_t)h;
  }
  
  int8_t b = pgm_read_byte(&m->b2[0]);
  acc += (int16_t)b * 256;
  
  // Деквантование выхода: int16_t → float
  float out_f = acc * m->w2_scale * (1.0f / 256.0f) * m->output_scale;
  
  return out_f;
}
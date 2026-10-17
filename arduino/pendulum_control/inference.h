#ifndef INFERENCE_H
#define INFERENCE_H

#include <Arduino.h>
#include <avr/pgmspace.h>

#include "config.h"
#include "model_schema.h"
#include "model_weights.h"  // задаёт NN_INPUT_DIM, NN_NUM_LAYERS, NN_MAX_DIM

// ============================================================================
// INT8 inference для MLP произвольной архитектуры.
// Активы хранятся в Q7.8 (int16), MAC — в int32, tanh — по таблице из
// model_weights.h. Нормализация входа берётся из статистики VecNormalize.
// ============================================================================

class QuantizedInference {
public:
  explicit QuantizedInference(const QuantizedModel* model);

  // Вход — сырые признаки в единицах обучения; нормализация внутри.
  float forward(const float input[NN_INPUT_DIM]);

private:
  const QuantizedModel* _model;

  // Два буфера переключаются между слоями, чтобы не копировать активации.
  int16_t _buf_a[NN_MAX_DIM];
  int16_t _buf_b[NN_MAX_DIM];

  static int16_t float_to_q78(float value);
  static int16_t tanh_q78(int16_t x_q78);
};

#endif // INFERENCE_H

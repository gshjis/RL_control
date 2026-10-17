#ifndef MODEL_SCHEMA_H
#define MODEL_SCHEMA_H

#include <stdint.h>

// ============================================================================
// Описание квантованной MLP. Архитектура задаётся сгенерированным
// model_weights.h (любое число слоёв и любой размер), здесь — только типы.
//
// Формат вычислений (см. tools/export_quantize.py):
//   * веса int8, симметричное квантование per-tensor;
//   * bias float32 в PROGMEM;
//   * активации int16 Q7.8 (значение = int / 256);
//   * tanh — по таблице g_tanh_lut.
// ============================================================================

#define NN_ACT_LINEAR 0
#define NN_ACT_TANH   1

struct QuantizedLayer {
  const int8_t* weights;   // [out_dim][in_dim], row-major, PROGMEM
  const float* bias;       // [out_dim], PROGMEM
  float weight_scale;      // scale симметричного квантования весов
  uint16_t in_dim;
  uint16_t out_dim;
  uint8_t activation;      // NN_ACT_LINEAR | NN_ACT_TANH
};

struct QuantizedModel {
  uint8_t num_layers;
  uint16_t input_dim;
  const QuantizedLayer* layers;  // PROGMEM
  const float* obs_mean;         // [input_dim], PROGMEM
  const float* obs_var;          // [input_dim], PROGMEM
  float obs_eps;
  float obs_clip;
};

#endif // MODEL_SCHEMA_H

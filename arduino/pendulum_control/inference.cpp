#include "inference.h"

#include <math.h>
#include <string.h>

// ── Перевод float → int16 Q7.8 с насыщением ───────────────────────────────
int16_t QuantizedInference::float_to_q78(float value) {
  float scaled = roundf(value * 256.0f);
  if (scaled > 32767.0f) return 32767;
  if (scaled < -32768.0f) return -32768;
  return (int16_t)scaled;
}

// ── Табличный tanh: 257 точек на [-8, 8], шаг 1/16 ────────────────────────
int16_t QuantizedInference::tanh_q78(int16_t x_q78) {
  // idx = (x + 8) * 16 = (x_q78 >> 4) + 128
  int16_t idx = (int16_t)((x_q78 >> 4) + 128);
  if (idx < 0) idx = 0;
  else if (idx > 256) idx = 256;
  return (int16_t)pgm_read_word(&g_tanh_lut[idx]);
}

QuantizedInference::QuantizedInference(const QuantizedModel* model)
    : _model(model) {}

float QuantizedInference::forward(const float input[NN_INPUT_DIM]) {
  const QuantizedModel* m = _model;

  // 1. Нормализация (статистика VecNormalize) и квантование входа в Q7.8.
  for (uint16_t i = 0; i < m->input_dim; i++) {
    float mean = pgm_read_float(&m->obs_mean[i]);
    float var = pgm_read_float(&m->obs_var[i]);
    float x = (input[i] - mean) / sqrtf(var + m->obs_eps);
    if (x > m->obs_clip) x = m->obs_clip;
    else if (x < -m->obs_clip) x = -m->obs_clip;
    _buf_a[i] = float_to_q78(x);
  }

  int16_t* in = _buf_a;
  int16_t* out = _buf_b;

  // 2. Слои. Дескриптор копируется из PROGMEM в стек.
  for (uint8_t l = 0; l < m->num_layers; l++) {
    QuantizedLayer layer;
    memcpy_P(&layer, &m->layers[l], sizeof(QuantizedLayer));

    for (uint16_t j = 0; j < layer.out_dim; j++) {
      const int8_t* row = layer.weights + (uint32_t)j * layer.in_dim;
      int32_t acc = 0;
      for (uint16_t i = 0; i < layer.in_dim; i++) {
        int16_t w = (int8_t)pgm_read_byte(&row[i]);
        acc += (int32_t)w * (int32_t)in[i];
      }

      float pre = (float)acc * layer.weight_scale / 256.0f
                  + pgm_read_float(&layer.bias[j]);

      if (layer.activation == NN_ACT_TANH) {
        out[j] = tanh_q78(float_to_q78(pre));
      } else {
        // Линейный слой — только выходной, одно значение; квантовать не нужно.
        return pre;
      }
    }

    int16_t* tmp = in;
    in = out;
    out = tmp;
  }

  // Линейного выходного слоя не было — возвращаем последнюю активацию.
  return (float)in[0] / 256.0f;
}

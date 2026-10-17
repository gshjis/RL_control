#ifndef INFERENCE_H
#define INFERENCE_H

#include <Arduino.h>
#include <avr/pgmspace.h>
#include "config.h"

// ============================================================================
// INT8 Inference Engine для MLP: 4 → 32 → 1 (Tanh)
// Fixed-point Q7.8 (int16_t для аккумуляции)
// ============================================================================

// Структура квантованных весов (в PROGMEM)
struct QuantizedModel {
  // Layer 1: Input(4) → Hidden(32)
  const int8_t* w1;        // [32 x 4] в PROGMEM
  const int8_t* b1;        // [32] в PROGMEM
  float w1_scale;          // Scale для весов L1
  float b1_scale;          // Scale для bias L1
  float input_scale;       // Scale для входа
  int8_t input_zp;         // Zero-point для входа
  int8_t hidden_zp;        // Zero-point для скрытого слоя
  
  // Layer 2: Hidden(32) → Output(1)
  const int8_t* w2;        // [1 x 32] в PROGMEM
  const int8_t* b2;        // [1] в PROGMEM
  float w2_scale;
  float b2_scale;
  float output_scale;      // Scale для выхода (обратно в float)
  int8_t output_zp;        // Zero-point для выхода
};

// Класс для инференса
class QuantizedInference {
public:
  QuantizedInference(const QuantizedModel* model);
  
  // Forward pass: float[4] вход → float выход
  // Вход в диапазоне, соответствующем обучению
  float forward(const float input[NN_INPUT_DIM]);
  
  // То же, но с уже квантованным входом (int8_t[4])
  float forwardQuantized(const int8_t input[NN_INPUT_DIM]);

private:
  const QuantizedModel* _model;
  
  // Буферы (в RAM, не в стеке)
  int16_t _hidden_acc[NN_HIDDEN_DIM];  // Аккумуляторы L1 (Q15.16)
  int8_t _hidden_out[NN_HIDDEN_DIM];   // Выход L1 после Tanh (Q7.8)
  
  // Tanh lookup table для Q7.8
  static int8_t tanh_q78(int16_t x_q78);
  
  // Умножение-аккумуляция с квантованием
  int16_t mac_q78(const int8_t* weights, const int8_t* input, int len, 
                  float w_scale, float in_scale, int8_t w_zp, int8_t in_zp, 
                  int8_t bias, float b_scale, int8_t out_zp);
};

#endif // INFERENCE_H
#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <avr/pgmspace.h>
#include "config.h"

// ============================================================================
// КВАНТОВАННЫЕ ВЕСА НС (INT8 asymmetric)
// Генерируется автоматически: python tools/export_quantize.py
// 
// Архитектура: Input(4) → Hidden(32, Tanh) → Output(1)
// ============================================================================

// --- Layer 1: 4 → 32 ---
// Weights: [32 x 4] = 128 int8_t
const int8_t g_w1[32 * 4] PROGMEM = {
  // TODO: Заполнить export_quantize.py
  // Нейрон 0
  0, 0, 0, 0,
  // Нейрон 1
  0, 0, 0, 0,
  // ... 32 нейрона
};

// Biases: [32] int8_t
const int8_t g_b1[32] PROGMEM = {
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
};

// --- Layer 2: 32 → 1 ---
// Weights: [1 x 32] = 32 int8_t
const int8_t g_w2[32] PROGMEM = {
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
};

// Biases: [1] int8_t
const int8_t g_b2[1] PROGMEM = { 0 };

// --- Quantization parameters (float) ---
// Масштабы и zero-points для де/квантования
struct QuantizedModel g_quantized_model = {
  // Layer 1
  .w1 = g_w1,
  .b1 = g_b1,
  .w1_scale = 0.01f,      // Пример: веса в [-1, 1] → int8_t
  .b1_scale = 0.01f,
  .input_scale = 1.0f,    // Вход уже в Q7.8
  .input_zp = 0,
  .hidden_zp = 0,
  
  // Layer 2
  .w2 = g_w2,
  .b2 = g_b2,
  .w2_scale = 0.01f,
  .b2_scale = 0.01f,
  .output_scale = 24.0f,  // Выход в Ньютонах (MAX_FORCE_N)
  .output_zp = 0,
};

#endif // MODEL_WEIGHTS_H
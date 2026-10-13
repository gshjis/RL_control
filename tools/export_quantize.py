#!/usr/bin/env python3
"""
Export + INT8 Quantization для PPO политики → Arduino model_weights.h

Usage:
    python tools/export_quantize.py \
        --model checkpoints/ppo/validation/model_3500000 \
        --output arduino/pendulum_control/model_weights.h
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Добавляем путь к пакетам
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.controllers.ppo.ppo import PPOController
from packages.controllers.ppo_config import PPOConfig
from packages.simulation.co.datatypes import ControllerConfig


def quantize_asymmetric(tensor: np.ndarray, bits: int = 8) -> tuple[np.ndarray, float, int]:
    """
    Asymmetric quantization: float -> int8
    
    Returns:
        quantized: int8 tensor
        scale: float scale factor
        zero_point: int zero point
    """
    qmin = -128 if bits == 8 else -(1 << (bits - 1))
    qmax = 127 if bits == 8 else (1 << (bits - 1)) - 1
    
    min_val = float(tensor.min())
    max_val = float(tensor.max())
    
    if max_val == min_val:
        scale = 1.0
        zero_point = 0
    else:
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(round(qmin - min_val / scale))
        zero_point = max(qmin, min(qmax, zero_point))
    
    quantized = np.clip(np.round(tensor / scale + zero_point), qmin, qmax).astype(np.int8)
    return quantized, scale, zero_point


def dequantize_asymmetric(quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Inverse: int8 -> float"""
    return (quantized.astype(np.float32) - zero_point) * scale


def export_model(
    model_path: str,
    output_path: str,
    max_force: float = 24.0,
    hidden_dim: int = 32,
) -> None:
    """Экспорт PPO модели в Arduino C header"""
    
    print(f"Loading model from {model_path}...")
    
    # Загружаем контроллер
    controller = PPOController.load(model_path)
    
    # Получаем веса политики (SB3 PPO policy)
    policy = controller._policy
    
    # Извлекаем веса MLP
    # SB3 MlpPolicy: features_extractor -> mlp_extractor -> action_net/value_net
    # Для PPO: policy.mlp_extractor.policy_net (Sequential)
    
    state_dict = policy.state_dict()
    
    # Ищем веса политики
    # Обычно: 'mlp_extractor.policy_net.0.weight', 'mlp_extractor.policy_net.0.bias', ...
    # 'action_net.weight', 'action_net.bias'
    
    print("State dict keys:")
    for k, v in state_dict.items():
        print(f"  {k}: {tuple(v.shape)}")
    
    # Предполагаем архитектуру: 4 -> 32 -> 32 -> 1 (или 4 -> 32 -> 1)
    # Ищем Linear слои
    
    # Пробуем разные варианты имен
    w1 = b1 = w2 = b2 = None
    
    # Вариант 1: mlp_extractor.policy_net
    if 'mlp_extractor.policy_net.0.weight' in state_dict:
        w1 = state_dict['mlp_extractor.policy_net.0.weight'].cpu().numpy().T  # [out, in] -> [in, out]
        b1 = state_dict['mlp_extractor.policy_net.0.bias'].cpu().numpy()
        w2 = state_dict['mlp_extractor.policy_net.2.weight'].cpu().numpy().T
        b2 = state_dict['mlp_extractor.policy_net.2.bias'].cpu().numpy()
    # Вариант 2: action_net напрямую
    elif 'action_net.weight' in state_dict:
        w1 = state_dict['action_net.weight'].cpu().numpy().T
        b1 = state_dict['action_net.bias'].cpu().numpy()
        # Если только один слой
        w2 = None
        b2 = None
    else:
        raise ValueError("Cannot find policy weights in state_dict")
    
    print(f"Layer 1: {w1.shape} -> {b1.shape}")
    if w2 is not None:
        print(f"Layer 2: {w2.shape} -> {b2.shape}")
    
    # Если 3 слоя (4->32->32->1), объединяем или берём последний
    # Для простоты: если w2 есть и это [32, 1], а w1 [4, 32] — ок
    # Если w2 [32, 32] — нужен третий слой
    
    # Квантование
    print("\nQuantizing...")
    
    # Layer 1
    w1_q, w1_scale, w1_zp = quantize_asymmetric(w1)
    b1_q, b1_scale, b1_zp = quantize_asymmetric(b1)
    
    # Layer 2
    if w2 is not None:
        w2_q, w2_scale, w2_zp = quantize_asymmetric(w2)
        b2_q, b2_scale, b2_zp = quantize_asymmetric(b2)
    else:
        # Single layer: выход напрямую
        w2_q = np.array([[1]], dtype=np.int8)  # identity
        b2_q = np.array([0], dtype=np.int8)
        w2_scale = 1.0
        b2_scale = 1.0
        w2_zp = 0
        b2_zp = 0
    
    # Input scale: вход в Q7.8 (features_q >> 3), масштаб = 1/256 * 32 = 1/8
    # Но проще: пусть вход уже в float, а квантование делается в forward
    input_scale = 1.0
    input_zp = 0
    hidden_zp = 0
    
    # Output scale: действия в [-max_force, max_force]
    output_scale = max_force
    output_zp = 0
    
    # Генерация C header
    print(f"\nGenerating {output_path}...")
    
    header = generate_header(
        w1_q, b1_q, w1_scale, b1_scale,
        w2_q, b2_q, w2_scale, b2_scale,
        input_scale, input_zp, hidden_zp,
        output_scale, output_zp,
        hidden_dim,
    )
    
    Path(output_path).write_text(header)
    print("Done!")
    
    # Валидация: сравниваем inference
    validate_quantization(
        w1, b1, w2, b2,
        w1_q, b1_q, w1_scale, b1_scale, w1_zp, b1_zp,
        w2_q, b2_q, w2_scale, b2_scale, w2_zp, b2_zp,
        input_scale, output_scale,
    )


def generate_header(
    w1_q, b1_q, w1_scale, b1_scale,
    w2_q, b2_q, w2_scale, b2_scale,
    input_scale, input_zp, hidden_zp,
    output_scale, output_zp,
    hidden_dim,
) -> str:
    """Генерация model_weights.h"""
    
    def array_to_c(arr, name, cols=8):
        lines = [f"const int8_t {name}[{arr.size}] PROGMEM = {{"]
        for i in range(0, len(arr), cols):
            chunk = arr[i:i+cols]
            lines.append("  " + ", ".join(str(int(x)) for x in chunk) + ",")
        lines.append("};")
        return "\n".join(lines)
    
    w1_flat = w1_q.flatten()
    w2_flat = w2_q.flatten()
    
    header = f'''#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <avr/pgmspace.h>
#include "config.h"

// ============================================================================
// КВАНТОВАННЫЕ ВЕСА НС (INT8 asymmetric)
// Автогенерировано: python tools/export_quantize.py
// Архитектура: Input(4) -> Hidden({hidden_dim}, Tanh) -> Output(1)
// ============================================================================

// --- Layer 1: 4 -> {hidden_dim} ---
{array_to_c(w1_flat, "g_w1")}

// Biases: [{hidden_dim}]
{array_to_c(b1_q, "g_b1")}

// --- Layer 2: {hidden_dim} -> 1 ---
{array_to_c(w2_flat, "g_w2")}

// Biases: [1]
{array_to_c(b2_q, "g_b2")}

// --- Quantization parameters ---
struct QuantizedModel g_quantized_model = {{
  // Layer 1
  .w1 = g_w1,
  .b1 = g_b1,
  .w1_scale = {w1_scale:.6f}f,
  .b1_scale = {b1_scale:.6f}f,
  .input_scale = {input_scale:.6f}f,
  .input_zp = {input_zp},
  .hidden_zp = {hidden_zp},
  
  // Layer 2
  .w2 = g_w2,
  .b2 = g_b2,
  .w2_scale = {w2_scale:.6f}f,
  .b2_scale = {b2_scale:.6f}f,
  .output_scale = {output_scale:.6f}f,
  .output_zp = {output_zp},
}};

#endif // MODEL_WEIGHTS_H
'''
    return header


def validate_quantization(
    w1, b1, w2, b2,
    w1_q, b1_q, w1_scale, b1_scale, w1_zp, b1_zp,
    w2_q, b2_q, w2_scale, b2_scale, w2_zp, b2_zp,
    input_scale, output_scale,
):
    """Проверка: квантованная модель ≈ оригинал"""
    print("\nValidating quantization...")
    
    # Тестовые входы
    np.random.seed(42)
    test_inputs = np.random.uniform(-1, 1, (100, 4)).astype(np.float32)
    
    # Оригинальный forward (float)
    def forward_float(x):
        h = np.tanh(x @ w1 + b1)
        if w2 is not None:
            y = h @ w2 + b2
        else:
            y = h  # single layer
        return y * output_scale
    
    # Квантованный forward (эмуляция Arduino)
    def forward_quant(x):
        # Квантуем вход
        x_q = np.clip(np.round(x * input_scale + 0), -128, 127).astype(np.int8)
        
        # L1
        h_acc = x_q.astype(np.int16) @ w1_q.astype(np.int16).T + b1_q.astype(np.int16) * 256
        h_q = np.tanh(h_acc / 256.0)  # Упрощённо
        h_q = np.clip(np.round(h_q * 256), -128, 127).astype(np.int8)
        
        # L2
        if w2 is not None:
            y_acc = h_q.astype(np.int16) @ w2_q.astype(np.int16).T + b2_q.astype(np.int16) * 256
        else:
            y_acc = h_q.astype(np.int16).sum(axis=1)  # Упрощение
        
        y = y_acc * w2_scale * (1/256.0) * output_scale
        return y
    
    y_float = forward_float(test_inputs)
    y_quant = forward_quant(test_inputs)
    
    mse = np.mean((y_float - y_quant) ** 2)
    max_err = np.max(np.abs(y_float - y_quant))
    
    print(f"  MSE: {mse:.6f}")
    print(f"  Max abs error: {max_err:.6f}")
    print(f"  Output range: [{y_float.min():.3f}, {y_float.max():.3f}] vs [{y_quant.min():.3f}, {y_quant.max():.3f}]")
    
    if max_err > 0.5:
        print("  WARNING: Large quantization error!")
    else:
        print("  OK: Quantization error acceptable")


def main():
    parser = argparse.ArgumentParser(description="Export PPO model to Arduino INT8")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (without .zip)")
    parser.add_argument("--output", required=True, help="Output .h file path")
    parser.add_argument("--max-force", type=float, default=24.0, help="Max force (N)")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer size")
    
    args = parser.parse_args()
    
    export_model(
        model_path=args.model,
        output_path=args.output,
        max_force=args.max_force,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()
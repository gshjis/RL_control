#!/usr/bin/env python3
"""Экспорт PPO-политики в INT8 C-заголовок для Arduino.

Архитектура берётся из самой модели: поддерживается любое число скрытых
слоёв и любой их размер. Формат вычислений описан в
``arduino/pendulum_control/model_schema.h``:

* веса — int8, симметричное квантование per-tensor (scale = max|w| / 127);
* смещения — float32 в PROGMEM: их мало, точность важнее экономии;
* активации (и вход) — int16 Q7.8, значение = int / 256;
* tanh — по таблице ``g_tanh_lut`` (257 точек на [-8, 8]);
* вход нормализуется статистикой VecNormalize, на которой обучалась модель.

Usage:
    python tools/export_quantize.py \
        --model checkpoints/ppo/validation/model_3500000 \
        --output arduino/pendulum_control/model_weights.h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Добавляем путь к пакетам
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.controllers.ppo.ppo import PPOController

Q_SHIFT = 8
Q_SCALE = float(1 << Q_SHIFT)
Q_MIN = -32768
Q_MAX = 32767

ACT_LINEAR = 0
ACT_TANH = 1

TANH_X_MAX = 8.0
TANH_LUT_STEP = 1.0 / 16.0
TANH_LUT_SIZE = 257


# ── Квантование ───────────────────────────────────────────────────────────
def build_tanh_lut() -> np.ndarray:
    """Возвращает таблицу tanh в Q7.8: 257 точек на [-8, 8] с шагом 1/16."""
    xs = np.linspace(-TANH_X_MAX, TANH_X_MAX, TANH_LUT_SIZE)
    return np.round(np.tanh(xs) * Q_SCALE).astype(np.int16)


def tanh_q78(x_q78: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Повторяет поиск из прошивки: idx = (x_q78 >> 4) + 128, клип [0, 256]."""
    idx = np.clip((x_q78.astype(np.int32) >> 4) + 128, 0, TANH_LUT_SIZE - 1)
    return lut[idx]


def quantize_symmetric(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    """Симметричное INT8-квантование per-tensor (zero point = 0)."""
    amax = float(np.max(np.abs(tensor))) if tensor.size else 0.0
    scale = amax / 127.0 if amax > 0.0 else 1.0
    quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return quantized, scale


def float_to_q78(values: np.ndarray) -> np.ndarray:
    """Переводит float в int16 Q7.8 с насыщением (как в прошивке)."""
    scaled = np.round(np.asarray(values, dtype=np.float64) * Q_SCALE)
    return np.clip(scaled, Q_MIN, Q_MAX).astype(np.int16)


# ── Извлечение слоёв ──────────────────────────────────────────────────────
def extract_layers(policy) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Возвращает слои политики: [(weights[out, in], bias[out], activation)]."""
    state = policy.state_dict()
    layers: list[tuple[np.ndarray, np.ndarray, int]] = []

    index = 0
    while f"mlp_extractor.policy_net.{index}.weight" in state:
        weight = state[f"mlp_extractor.policy_net.{index}.weight"]
        bias = state[f"mlp_extractor.policy_net.{index}.bias"]
        layers.append(
            (
                weight.detach().cpu().numpy(),
                bias.detach().cpu().numpy(),
                ACT_TANH,
            )
        )
        index += 2

    if "action_net.weight" not in state:
        raise ValueError("В state_dict нет action_net: политика не MLP?")

    layers.append(
        (
            state["action_net.weight"].detach().cpu().numpy(),
            state["action_net.bias"].detach().cpu().numpy(),
            ACT_LINEAR,
        )
    )
    return layers


def normalization_stats(
    controller: PPOController, input_dim: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Возвращает (mean, var, eps, clip) статистики VecNormalize."""
    vec = controller.vec_normalize
    obs_rms = getattr(vec, "obs_rms", None) if vec is not None else None
    if vec is None or not hasattr(obs_rms, "mean"):
        print("WARNING: VecNormalize не найден — вход не будет нормализован")
        return (
            np.zeros(input_dim, dtype=np.float64),
            np.ones(input_dim, dtype=np.float64),
            1e-8,
            10.0,
        )

    mean = np.asarray(obs_rms.mean, dtype=np.float64).reshape(-1)
    var = np.asarray(obs_rms.var, dtype=np.float64).reshape(-1)
    if mean.size != input_dim:
        raise ValueError(
            f"Статистика нормализации имеет {mean.size} компонент, "
            f"а модель ждёт {input_dim}"
        )
    return mean, var, float(vec.epsilon), float(vec.clip_obs)


# ── Эмуляция forward ──────────────────────────────────────────────────────
def forward_float(
    raw: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray, int]],
    mean: np.ndarray,
    var: np.ndarray,
    eps: float,
    clip: float,
) -> np.ndarray:
    """Эталон: обычный forward во float с нормализованным входом."""
    x = np.clip((raw - mean) / np.sqrt(var + eps), -clip, clip)
    for weight, bias, activation in layers:
        x = x @ weight.T + bias
        if activation == ACT_TANH:
            x = np.tanh(x)
    return x[:, 0]


def forward_quantized(
    raw: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray, int]],
    lut: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    eps: float,
    clip: float,
) -> np.ndarray:
    """Эталон прошивки: нормализация + Q7.8 + int32 MAC + табличный tanh."""
    normalized = np.clip((raw - mean) / np.sqrt(var + eps), -clip, clip)
    activations = float_to_q78(normalized)

    for weight, bias, activation in layers:
        weight_q, scale = quantize_symmetric(weight)
        acc = activations.astype(np.int32) @ weight_q.astype(np.int32).T
        pre = acc * scale / Q_SCALE + bias.astype(np.float64)
        if activation == ACT_TANH:
            activations = tanh_q78(float_to_q78(pre), lut)
        else:
            return pre[:, 0]
    return activations[:, 0] / Q_SCALE


# ── Генерация заголовка ───────────────────────────────────────────────────
def _c_float(value: float) -> str:
    """Форматирует float как валидный C++-литерал (0f и 1f невалидны)."""
    text = f"{float(value):.8g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"


def _array_to_c(
    values: np.ndarray, c_type: str, name: str, per_line: int = 12
) -> str:
    """Форматирует одномерный массив как C-инициализатор в PROGMEM."""
    rendered = []
    for value in values.reshape(-1):
        if c_type == "float":
            rendered.append(_c_float(value))
        else:
            rendered.append(str(int(value)))

    lines = [f"const {c_type} {name}[{len(rendered)}] PROGMEM = {{"]
    for start in range(0, len(rendered), per_line):
        chunk = ", ".join(rendered[start : start + per_line])
        lines.append(f"  {chunk},")
    lines.append("};")
    return "\n".join(lines)


def generate_header(
    layers: list[tuple[np.ndarray, np.ndarray, int]],
    lut: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    eps: float,
    clip: float,
    source: str,
) -> str:
    """Собирает model_weights.h для произвольной MLP-архитектуры."""
    input_dim = int(layers[0][0].shape[1])
    dims = [input_dim] + [int(weight.shape[0]) for weight, _, _ in layers]
    max_dim = max(dims)

    blocks: list[str] = []
    entries: list[str] = []
    for index, (weight, bias, activation) in enumerate(layers):
        weight_q, scale = quantize_symmetric(weight)
        in_dim, out_dim = int(weight.shape[1]), int(weight.shape[0])
        blocks.append(
            f"// --- Слой {index + 1}: {in_dim} -> {out_dim} ---\n"
            + _array_to_c(weight_q, "int8_t", f"g_w{index}")
            + "\n\n"
            + _array_to_c(bias, "float", f"g_b{index}")
        )
        act_name = "NN_ACT_TANH" if activation == ACT_TANH else "NN_ACT_LINEAR"
        entries.append(
            f"  {{ g_w{index}, g_b{index}, {_c_float(scale)}, {in_dim}, "
            f"{out_dim}, {act_name} }},"
        )

    architecture = " -> ".join(str(dim) for dim in dims)
    body = "\n\n".join(blocks)
    entries_text = "\n".join(entries)

    return f"""#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <avr/pgmspace.h>

#include "config.h"
#include "model_schema.h"

// ============================================================================
// КВАНТОВАННЫЕ ВЕСА НС (INT8, симметричное квантование)
// Автогенерировано из: {source}
// Архитектура: {architecture}
// ============================================================================

#define NN_NUM_LAYERS {len(layers)}
#define NN_INPUT_DIM {input_dim}
#define NN_MAX_DIM {max_dim}

#define NN_OBS_EPS {_c_float(eps)}
#define NN_OBS_CLIP {_c_float(clip)}

{body}

// --- Таблица tanh (Q7.8), 257 точек на [-8, 8] ---
{_array_to_c(lut, "int16_t", "g_tanh_lut", 16)}

// --- Нормализация входа (статистика VecNormalize) ---
{_array_to_c(mean, "float", "g_obs_mean", 8)}

{_array_to_c(var, "float", "g_obs_var", 8)}

// --- Описание слоёв ---
const QuantizedLayer g_layers[NN_NUM_LAYERS] PROGMEM = {{
{entries_text}
}};

const QuantizedModel g_quantized_model = {{
  NN_NUM_LAYERS,
  NN_INPUT_DIM,
  g_layers,
  g_obs_mean,
  g_obs_var,
  NN_OBS_EPS,
  NN_OBS_CLIP,
}};

#endif // MODEL_WEIGHTS_H
"""


# ── Основной сценарий ─────────────────────────────────────────────────────
def export_model(model_path: str, output_path: str) -> None:
    """Квантует политику из чекпоинта и пишет C-заголовок."""
    print(f"Loading model from {model_path}...")
    controller = PPOController.load(model_path)
    if controller.model is None:
        raise RuntimeError(f"Модель не загружена: {model_path}")

    layers = extract_layers(controller.model.policy)
    input_dim = int(layers[0][0].shape[1])
    mean, var, eps, clip = normalization_stats(controller, input_dim)
    lut = build_tanh_lut()

    print("Architecture:")
    previous = input_dim
    for weight, _, activation in layers:
        out_dim = int(weight.shape[0])
        name = "tanh" if activation == ACT_TANH else "linear"
        print(f"  {previous} -> {out_dim} ({name})")
        previous = out_dim

    for weight, _, _ in layers:
        amax = float(np.max(np.abs(weight)))
        acc_max = 127.0 * Q_MAX * int(weight.shape[1])
        print(f"  weights max |w| = {amax:.6f}, worst-case |acc| = {acc_max:.3e}")
        if acc_max >= 2**31:
            raise ValueError("Насыщение int32 в аккумуляторе: слишком широкий слой")

    print("\nValidating quantization...")
    rng = np.random.default_rng(0)
    raw = rng.uniform(-3.0, 3.0, (200, input_dim))
    reference = forward_float(raw, layers, mean, var, eps, clip)
    quantized = forward_quantized(raw, layers, lut, mean, var, eps, clip)

    mse = float(np.mean((reference - quantized) ** 2))
    max_error = float(np.max(np.abs(reference - quantized)))
    print(f"  MSE: {mse:.6f}")
    print(f"  Max abs error: {max_error:.6f}")
    print(
        f"  Output range: [{reference.min():.3f}, {reference.max():.3f}] "
        f"vs [{quantized.min():.3f}, {quantized.max():.3f}]"
    )
    if max_error > 0.5:
        print("  WARNING: Large quantization error!")
    else:
        print("  OK: Quantization error acceptable")

    header = generate_header(
        layers=layers,
        lut=lut,
        mean=mean,
        var=var,
        eps=eps,
        clip=clip,
        source=model_path,
    )
    Path(output_path).write_text(header, encoding="utf-8")
    print(f"\nGenerated {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PPO model to Arduino INT8")
    parser.add_argument(
        "--model", required=True, help="Path to model checkpoint (without .zip)"
    )
    parser.add_argument("--output", required=True, help="Output .h file path")
    args = parser.parse_args()

    export_model(model_path=args.model, output_path=args.output)


if __name__ == "__main__":
    main()

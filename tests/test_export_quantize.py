"""Тесты экспортёра INT8: валидные литералы и близость к float-эталону."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_quantize", ROOT / "tools" / "export_quantize.py"
)
assert SPEC is not None and SPEC.loader is not None
export_quantize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_quantize)

MEAN = np.array([0.1, -0.2, 0.05, 0.0])
VAR = np.array([1.5, 0.8, 2.0, 0.5])
EPS = 1e-8
CLIP = 10.0


def _make_layers(arch: list[int], seed: int = 7):
    """Собирает случайную MLP: 4 -> arch... -> 1."""
    rng = np.random.default_rng(seed)
    dims = [4] + arch + [1]
    layers = []
    for i in range(len(dims) - 1):
        weight = rng.normal(0, 1.0 / np.sqrt(dims[i]), (dims[i + 1], dims[i]))
        bias = rng.normal(0, 0.1, dims[i + 1])
        activation = (
            export_quantize.ACT_TANH
            if i < len(dims) - 2
            else export_quantize.ACT_LINEAR
        )
        layers.append((weight, bias, activation))
    return layers


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "0.0f"), (1.0, "1.0f"), (-2.0, "-2.0f"), (1e-8, "1e-08f")],
)
def test_c_float_literals_are_valid(value: float, expected: str) -> None:
    """0f и 1f не компилируются в C++, поэтому нужна дробная точка."""
    assert export_quantize._c_float(value) == expected


def test_header_has_no_invalid_float_suffixes() -> None:
    """В сгенерированном заголовке нет литералов вида 0f / 1f."""
    layers = _make_layers([8, 6])
    lut = export_quantize.build_tanh_lut()
    header = export_quantize.generate_header(
        layers, lut, MEAN, VAR, EPS, CLIP, "test"
    )

    assert re.search(r"(?<![\d.])[01]f\b", header) is None
    assert "#define NN_NUM_LAYERS 3" in header
    assert "#define NN_INPUT_DIM 4" in header
    assert "#define NN_MAX_DIM 8" in header


@pytest.mark.parametrize("arch", ([8, 6], [], [32]))
def test_quantized_forward_is_close_to_float(arch: list[int]) -> None:
    """Эмуляция прошивки отличается от float-forward незначительно."""
    layers = _make_layers(arch)
    lut = export_quantize.build_tanh_lut()
    raw = np.random.default_rng(11).uniform(-3.0, 3.0, (200, 4))

    reference = export_quantize.forward_float(raw, layers, MEAN, VAR, EPS, CLIP)
    quantized = export_quantize.forward_quantized(
        raw, layers, lut, MEAN, VAR, EPS, CLIP
    )

    assert np.max(np.abs(reference - quantized)) < 0.15

"""Разметка векторов состояния: телеметрия датчика, цель и наблюдение.

Единственное место, где зафиксирован порядок компонент. Раньше индексы
``state[1]`` / ``state[6]`` и ``COS_INDEX = 1`` были размазаны по пяти
модулям и тестам, а сам порядок описан только комментарием в
``co_sensor.cpp``.

Форматы:

* **телеметрия** (8 каналов, ``SensorBlock::get_telemetry``):
  ``[x, cos t1, sin t1, cos t2, sin t2, dx, dt1, dt2]``
* **цель** (6 компонент): ``[x, cos t1, sin t1, cos t2, sin t2, dx]``
* **наблюдение среды**: телеметрия ‖ цель = 14 компонент
* **чистое состояние** (``ObjectOfControl.get_clean_state``):
  ``[x, t1, t2, dx, dt1, dt2]`` — углы в радианах, без тригонометрии
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Телеметрия датчика ────────────────────────────────────────────────────
TELEMETRY_DIM = 8
IDX_X = 0
IDX_COS_T1 = 1
IDX_SIN_T1 = 2
IDX_COS_T2 = 3
IDX_SIN_T2 = 4
IDX_DX = 5
IDX_DTH1 = 6
IDX_DTH2 = 7

TELEMETRY_LABELS = (
    "x",
    "cos(theta1)",
    "sin(theta1)",
    "cos(theta2)",
    "sin(theta2)",
    "dx/dt",
    "dtheta1/dt",
    "dtheta2/dt",
)

# ── Цель ──────────────────────────────────────────────────────────────────
TARGET_DIM = 6

# ── Наблюдение среды = телеметрия ‖ цель ──────────────────────────────────
OBS_DIM = TELEMETRY_DIM + TARGET_DIM
TARGET_SLICE = slice(TELEMETRY_DIM, OBS_DIM)

# ── Чистое состояние (истинная физика, без шума датчика) ──────────────────
CLEAN_STATE_DIM = 6
CLEAN_IDX_X = 0
CLEAN_IDX_T1 = 1
CLEAN_IDX_T2 = 2
CLEAN_IDX_DX = 3
CLEAN_IDX_DTH1 = 4
CLEAN_IDX_DTH2 = 5

CLEAN_STATE_LABELS = (
    "x",
    "theta1",
    "theta2",
    "dx/dt",
    "dtheta1/dt",
    "dtheta2/dt",
)


@dataclass(frozen=True)
class Telemetry:
    """Разобранные компоненты телеметрии датчика."""

    x: float
    cos_t1: float
    sin_t1: float
    cos_t2: float
    sin_t2: float
    dx: float
    dtheta1: float
    dtheta2: float

    @property
    def theta1(self) -> float:
        """Возвращает угол первого звена в радианах (atan2, диапазон [-pi, pi])."""
        return float(np.arctan2(self.sin_t1, self.cos_t1))

    @property
    def theta2(self) -> float:
        """Возвращает угол второго звена в радианах (atan2)."""
        return float(np.arctan2(self.sin_t2, self.cos_t2))


def unpack_telemetry(values: np.ndarray) -> Telemetry:
    """Принимает вектор телеметрии (или наблюдение), возвращает разбор."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape[0] < TELEMETRY_DIM:
        raise ValueError(
            f"ожидалось не меньше {TELEMETRY_DIM} компонент, получено {array.shape[0]}"
        )
    return Telemetry(*(float(component) for component in array[:TELEMETRY_DIM]))


def split_observation(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Принимает наблюдение среды, возвращает (телеметрия, цель)."""
    array = np.asarray(observation, dtype=np.float64).reshape(-1)
    if array.shape[0] != OBS_DIM:
        raise ValueError(
            f"наблюдение должно иметь {OBS_DIM} компонент, получено {array.shape[0]}"
        )
    return array[:TELEMETRY_DIM], array[TARGET_SLICE]


__all__ = [
    "CLEAN_IDX_DX",
    "CLEAN_IDX_DTH1",
    "CLEAN_IDX_DTH2",
    "CLEAN_IDX_T1",
    "CLEAN_IDX_T2",
    "CLEAN_IDX_X",
    "CLEAN_STATE_DIM",
    "CLEAN_STATE_LABELS",
    "IDX_COS_T1",
    "IDX_COS_T2",
    "IDX_DX",
    "IDX_DTH1",
    "IDX_DTH2",
    "IDX_SIN_T1",
    "IDX_SIN_T2",
    "IDX_X",
    "OBS_DIM",
    "TARGET_DIM",
    "TARGET_SLICE",
    "TELEMETRY_DIM",
    "TELEMETRY_LABELS",
    "Telemetry",
    "split_observation",
    "unpack_telemetry",
]

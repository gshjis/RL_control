from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from packages.simulation.co.datatypes import ControllerConfig, PlantConfig
from packages.simulation.co.layout import (
    IDX_COS_T1,
    IDX_COS_T2,
    IDX_DTH1,
    IDX_DTH2,
    IDX_DX,
    IDX_SIN_T1,
    IDX_SIN_T2,
    IDX_X,
    TELEMETRY_DIM,
)


def build_features_auto(
    plant: PlantConfig,
    controller: ControllerConfig,
    telemetry: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Автоматическая архитектура: 4/5/7/8 признаков.

    Зависит от конфигурации:
    - single_pendulum_mode=True, position_controlled=False: 4 (sinθ1, cosθ1, θ̇1, ẋ)
    - single_pendulum_mode=True, position_controlled=True:  5 (+ x - x_target)
    - single_pendulum_mode=False, position_controlled=False: 7 (+ sinθ2, cosθ2, θ̇2)
    - single_pendulum_mode=False, position_controlled=True:  8 (+ x - x_target)
    """
    if telemetry.shape[0] < TELEMETRY_DIM:
        raise ValueError(f"telemetry must have at least {TELEMETRY_DIM} components")
    if target.shape[0] < 1:
        raise ValueError("target must have at least x_target component")

    # Базовые признаки первого звена (всегда)
    sin_t1 = float(telemetry[IDX_SIN_T1])
    cos_t1 = float(telemetry[IDX_COS_T1])
    dth1 = float(telemetry[IDX_DTH1])
    dx = float(telemetry[IDX_DX])

    features = [sin_t1, cos_t1, dth1, dx]

    # Второе звено (если не single_pendulum_mode) — ДО ошибки позиции
    if not plant.single_pendulum_mode:
        sin_t2 = float(telemetry[IDX_SIN_T2])
        cos_t2 = float(telemetry[IDX_COS_T2])
        dth2 = float(telemetry[IDX_DTH2])
        features.extend([sin_t2, cos_t2, dth2])

    # Ошибка позиции тележки (для позиционного управления) — в конце
    if controller.position_controlled:
        x = float(telemetry[IDX_X])
        x_target = float(target[0])
        features.append(x - x_target)

    return np.asarray(features, dtype=np.float64)


def get_obs_dim(plant: PlantConfig, controller: ControllerConfig) -> int:
    """Автоматическая размерность наблюдения: 4/5/7/8 признаков."""
    dim = 4
    if controller.position_controlled:
        dim += 1
    if not plant.single_pendulum_mode:
        dim += 3
    return dim


# ── Legacy FeatureConfig (для совместимости с тестами feature-ветки) ────────


@dataclass(frozen=True)
class FeatureConfig:
    n_links: int = field(default=1)
    position_controlled: bool = field(default=False)

    @property
    def dim(self) -> int:
        return 3 * self.n_links + 2 + self.position_controlled


def build_features(
    telemetry: np.ndarray, target: np.ndarray, config: FeatureConfig
) -> np.ndarray:
    """Legacy функция для совместимости с тестами.

    Преобразует FeatureConfig в эквивалентные PlantConfig/ControllerConfig
    и вызывает build_features_auto.
    """
    plant = PlantConfig(single_pendulum_mode=(config.n_links == 1))
    controller = ControllerConfig(position_controlled=config.position_controlled)
    return build_features_auto(plant, controller, telemetry, target)


__all__ = [
    "FeatureConfig",
    "build_features",
    "build_features_auto",
    "get_obs_dim",
]

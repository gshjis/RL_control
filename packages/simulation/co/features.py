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
    """TEMPORARY: Old 14-input architecture (telemetry 8 + target 6).

    Original 14-dim observation:
    - telemetry (8): [x, cosθ1, sinθ1, cosθ2, sinθ2, ẋ, θ̇1, θ̇2]
    - target (6):  [x_t, cosθ1_t, sinθ1_t, cosθ2_t, sinθ2_t, ẋ_t]
    """
    if telemetry.shape[0] < TELEMETRY_DIM:
        raise ValueError(f"telemetry must have at least {TELEMETRY_DIM} components")
    if target.shape[0] < 6:
        raise ValueError("target must have at least 6 components for 14-dim obs")

    # Concatenate telemetry (8) + target (6) = 14
    features = np.concatenate([
        telemetry[:TELEMETRY_DIM],      # 8 values
        target[:6],                      # 6 values
    ]).astype(np.float64)

    return features


def _build_features_auto_new(
    plant: PlantConfig,
    controller: ControllerConfig,
    telemetry: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Новая архитектура: 4/5/7/8 признаков (sin/cos + velocities + pos error)."""
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
    """TEMPORARY: Returns 14 for old architecture (telemetry 8 + target 6)."""
    # Old 14-dim: 8 telemetry + 6 target
    return 14


def _get_obs_dim_new(plant: PlantConfig, controller: ControllerConfig) -> int:
    """Новая архитектура: 4/5/7/8 признаков."""
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

from __future__ import annotations

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
    """Автоматически строит вектор признаков для НС из телеметрии и цели.

    4 кейса на основе конфигурации:
    - single_pendulum_mode + position_controlled=False: 4 признака
      [sinθ1, cosθ1, dθ1, dx]
    - single_pendulum_mode + position_controlled=True: 5 признаков
      [sinθ1, cosθ1, dθ1, dx, x - x_target]
    - double pendulum + position_controlled=False: 7 признаков
      [sinθ1, cosθ1, dθ1, dx, sinθ2, cosθ2, dθ2]
    - double pendulum + position_controlled=True: 8 признаков
      [sinθ1, cosθ1, dθ1, dx, sinθ2, cosθ2, dθ2, x - x_target]

    Абсолютная позиция тележки (x) НЕ включается — только dx и ошибка позиции.
    Это обеспечивает переносимость политики на реальную аппаратуру.
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

    # Ошибка позиции тележки (для позиционного управления)
    if controller.position_controlled:
        x = float(telemetry[IDX_X])
        x_target = float(target[0])
        features.append(x - x_target)

    # Второе звено (если не single_pendulum_mode)
    if not plant.single_pendulum_mode:
        sin_t2 = float(telemetry[IDX_SIN_T2])
        cos_t2 = float(telemetry[IDX_COS_T2])
        dth2 = float(telemetry[IDX_DTH2])
        features.extend([sin_t2, cos_t2, dth2])

    return np.asarray(features, dtype=np.float64)


def get_obs_dim(plant: PlantConfig, controller: ControllerConfig) -> int:
    """Возвращает размерность наблюдения для заданных конфигов."""
    # 4 базовых (sinθ1, cosθ1, dθ1, dx)
    dim = 4
    # +1 если позиционное управление
    if controller.position_controlled:
        dim += 1
    # +3 если двухзвенный
    if not plant.single_pendulum_mode:
        dim += 3
    return dim


# ── Legacy FeatureConfig (для совместимости с тестами feature-ветки) ────────
from dataclasses import dataclass, field


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
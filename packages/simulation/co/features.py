from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

# Назнааечение: настройка архитектуры НС
@dataclass(frozen=True)
class FeatureConfig:
    n_links: int = field(default=1)
    position_controlled: bool = field(default=False)

    @property
    def dim(self) -> int:
        return 3 * self.n_links + 2 + self.position_controlled


def build_features(telemetry: np.ndarray, target: np.ndarray, config: FeatureConfig) -> np.ndarray:
    # Предполагается, что мы работаем с 2/4/5/8 против 14 по умолчанию
    x, cos_theta1, sin_theta1, cos_theta2, sin_theta2, dx, dtheta1, dtheta2 = telemetry
    x_target = target[0]  # цель по x

    features = []
    features.append(cos_theta1)
    features.append(sin_theta1)
    features.append(dtheta1)
    features.append(x)
    features.append(dx)

    if config.position_controlled:
        features.append(x - x_target)

    if config.n_links == 2:
        features.append(cos_theta2)
        features.append(sin_theta2)
        features.append(dtheta2)

    return np.array(features, dtype=np.float64)  # features.tensor

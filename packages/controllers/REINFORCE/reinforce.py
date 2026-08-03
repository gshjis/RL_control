from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional  # noqa: UP035

import numpy as np
import torch
from controller import Controller
from datatypes import ControllerConfig
from loggers import Logger
from torch import nn

from packages.controllers.REINFORCE.mode_config import ReinforceNetworkConfig
from packages.simulation.CO import (
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorConfig,
)


def default_terminate_condition(state: ObjectOfControl) -> bool:
    """Падение если маятник отклонился от π более чем на 40°."""
    ...


class ReinforceNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list[int],
        activation: str = "tanh",
        output_activation: str = "tanh",
    ) -> None:
        ...

    def _get_activation(self, name: str) -> nn.Module:
        ...

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class Reinforce(Controller):
    def __init__(
        self, 
        config: ReinforceNetworkConfig,
        controller_config: ControllerConfig
    ) -> None:
        ...

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def get_action(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
        ...

    def save(self, path: str | Path) -> None:
        """Сохранить веса нейросети в файл."""
        ...

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        config: ReinforceNetworkConfig,
        controller_config: ControllerConfig,
    ) -> Reinforce:
        """Создать агента с конфигурацией и сразу загрузить веса."""
        ...

    def load(self, path: str | Path) -> None:
        """Загрузить веса нейросети из файла."""
        ...

    def reset(self) -> None:
        ...

    def train(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        target_state: np.ndarray,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
        episode_max_time: float = 150.0,
        epochs: int = 1000,
        episodes_per_epoch: int = 100,
        logger: Optional[Logger] = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:
        ...

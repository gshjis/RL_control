from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from base import Controller
from datatypes import ControllerConfig
from loggers import Logger
from torch import nn

from packages.controllers.reinforce.mode_config import ReinforceNetworkConfig
from packages.simulation.co import (
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorConfig,
)


def default_terminate_condition(state: ObjectOfControl) -> bool:
    """Принимает объект управления, возвращает признак отклонения от вертикали."""
    return abs(state.q[1] - np.pi) > np.radians(40)


class ReinforceNet(nn.Module):
    """Нейросеть REINFORCE для распределения действий и оценки масштаба."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list[int],
        activation: str = "tanh",
        output_activation: str = "tanh",
    ) -> None:
        """Принимает размеры слоёв и активации, строит нейросеть."""
        raise NotImplementedError("Сеть REINFORCE ещё не реализована.")

    def _get_activation(self, name: str) -> nn.Module:
        """Принимает имя активации, возвращает соответствующий модуль."""
        raise NotImplementedError("Активации REINFORCE ещё не реализованы.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Принимает тензор состояний, возвращает параметры распределения действий."""
        raise NotImplementedError("Прямой проход REINFORCE ещё не реализован.")


class Reinforce(Controller):
    """Контроллер и агент обучения методом REINFORCE."""

    def __init__(
        self, config: ReinforceNetworkConfig, controller_config: ControllerConfig
    ) -> None:
        """Принимает конфигурации сети и контроллера, создаёт агента."""
        raise NotImplementedError("Контроллер REINFORCE ещё не реализован.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Принимает тензор состояния, возвращает выходы сети агента."""
        raise NotImplementedError("Прямой проход агента ещё не реализован.")

    def get_action(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
        """Принимает состояние и цель, возвращает выбранное действие."""
        raise NotImplementedError("Выбор действия REINFORCE ещё не реализован.")

    def save(self, path: str | Path) -> None:
        """Сохранить веса нейросети в файл."""
        raise NotImplementedError("Сохранение REINFORCE ещё не реализовано.")

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        config: ReinforceNetworkConfig,
        controller_config: ControllerConfig,
    ) -> Reinforce:
        """Создать агента с конфигурацией и сразу загрузить веса."""
        raise NotImplementedError("Загрузка REINFORCE ещё не реализована.")

    def load(self, path: str | Path) -> None:
        """Загрузить веса нейросети из файла."""
        raise NotImplementedError("Загрузка REINFORCE ещё не реализована.")

    def reset(self) -> None:
        """Сбрасывает внутреннее состояние агента."""
        raise NotImplementedError("Сброс REINFORCE ещё не реализован.")

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
        logger: Logger | None = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:

        print(12)
        """Принимает параметры среды и обучения, запускает обучение агента."""
        raise NotImplementedError("Обучение REINFORCE ещё не реализовано.")

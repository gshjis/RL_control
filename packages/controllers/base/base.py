"""Абстрактный базовый контроллер."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from packages.simulation.co.datatypes import ControllerConfig


class Controller(ABC):
    """Абстрактный контроллер: читает наблюдение среды, возвращает действие.

    Единственный метод, который обязан реализовать наследник, — ``act``.
    Обучение к контракту не относится: им занимается слой тренера
    (см. ``packages.controllers.ppo.trainer.PPOTrainer``).
    """

    def __init__(self, controller_config: ControllerConfig) -> None:
        """Принимает конфигурацию контроллера."""
        self._name: str = self.__class__.__name__
        self._controller_config = controller_config

    def __str__(self) -> str:
        """Возвращает строковое представление контроллера."""
        return self._name

    def __repr__(self) -> str:
        """Возвращает представление контроллера с конфигурацией."""
        return f"{self.__class__.__name__}(config={self._controller_config})"

    def __getattr__(self, item: str) -> Any:
        """Проксирует обращение к полям конфигурации контроллера.

        Обращения к приватным полям идёт в обход (иначе — AttributeError),
        чтобы не зациклиться на этапе инициализации.
        """
        if item.startswith("_") or "_controller_config" not in self.__dict__:
            raise AttributeError(item)
        return getattr(self._controller_config, item)

    @property
    def name(self) -> str:
        """Возвращает имя контроллера."""
        return self._name

    @property
    def controller_config(self) -> ControllerConfig:
        """Возвращает конфигурацию контроллера."""
        return self._controller_config

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Принимает наблюдение среды, возвращает действие."""
        raise NotImplementedError

    def close(self) -> None:
        """Освобождает ресурсы. По умолчанию ничего не делает."""

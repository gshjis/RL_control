from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from packages.simulation.co.datatypes import ControllerConfig


class Controller(ABC):
    """Определяет общий интерфейс контроллеров объекта управления."""

    def __init__(self, config: ControllerConfig) -> None:
        """Принимает конфигурацию и сохраняет ограничения контроллера."""

        self._dt = config.dt
        self._max_force = config.max_force
        self._has_vel = config.has_velocity_sensors
        self._diff_cutoff = config.differentiator_cutoff_hz
        self._filter_cutoff = config.filter_cutoff_hz

        self.name: str
        self._last_action: float = 0.0

    @abstractmethod
    def action(self, state_target: np.ndarray) -> np.ndarray:
        """Принимает состояние и цель, возвращает управляющее действие."""
        ...

    @property
    def last_control_action(self) -> float:
        """Возвращает последнее управляющее воздействие."""
        return self._last_control_action

    @property
    def dt(self) -> float:
        """Возвращает шаг контроллера в секундах."""
        return self._dt

    def reset(self) -> None:
        """Сбрасывает сохранённое управляющее воздействие."""
        self._last_control_action = 0.0

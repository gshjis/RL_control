from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from loggers import Logger

from packages.controllers.PID.pid import PIDController
from packages.simulation.CO import (
    Controller,
    ControllerConfig,
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorConfig,
)


def check_linerised_position(s_clean: np.ndarray) -> bool:
    """Принимает состояние, возвращает близость к вертикали."""
    return abs(np.pi - s_clean[1]) < np.radians(50)


class SwingUp(Controller):
    """Раскачивает маятник энергетическим законом управления."""

    def __init__(
        self, config: ControllerConfig, K: float, plant_config: PlantConfig
    ) -> None:
        """Принимает конфигурацию, усиление и параметры растения."""
        super().__init__(config)
        self.name = "SwingUp"
        self._K = K
        self._plant_config = plant_config.copy()

    def get_control(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
        """Принимает состояние и цель, возвращает энергоуправляющую силу."""
        # Текущая энергия маятника
        E = (
            0.5 * self._plant_config.m1 * self._plant_config.L1 * (s_clean[4]) ** 2
            - self._plant_config.m1
            * self._plant_config.g
            * self._plant_config.L1
            * (1 - np.cos(s_clean[1]))
        )
        # Целевая энергия (верхнее положение)
        E_t = -2 * self._plant_config.m1 * self._plant_config.g * self._plant_config.L1
        return float(
            self._K * np.tanh(E - E_t) * np.sign(s_clean[4] * np.cos(s_clean[1]))
        )

    def train(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        target_state: np.ndarray,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
        episode_max_time: float = 150.0,
        logger: None | Logger = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:
        """SwingUp не обучается — управление задаётся аналитически."""
        raise NotImplementedError("SwingUp не требует обучения.")


class SwingUpAndBalance(Controller):
    """Переключает раскачку SwingUp и балансировку PID."""

    def __init__(
        self,
        config: ControllerConfig,
        swingup_controller: SwingUp,
        balance_controller: PIDController,
    ) -> None:
        """Принимает конфигурацию, контроллер раскачки и PID-балансировщик."""
        super().__init__(config)
        self.name = "BEAST"
        self._swing_up_controller = swingup_controller
        self._balance_controller = balance_controller

    def get_control(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
        """Принимает состояние и цель, возвращает силу выбранного контроллера."""
        if check_linerised_position(s_clean):
            return self._balance_controller.get_control(s_clean, target_state)
        return self._swing_up_controller.get_control(s_clean, target_state)

    def train(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        target_state: np.ndarray,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
        episode_max_time: float = 150.0,
        logger: None | Logger = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:
        """Композитный контроллер не обучается напрямую."""
        raise NotImplementedError("SwingUpAndBalance не требует обучения.")

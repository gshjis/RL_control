from __future__ import annotations

from typing import Any, Callable, Optional

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
    """Проверить, находится ли маятник вблизи вертикального положения.

    Возвращает ``True``, если отклонение от вертикали (π) меньше 50°.
    """
    return abs(np.pi - s_clean[1]) < np.radians(50)


class SwingUp(Controller):
    """Энергетический контроллер раскачки маятника из нижнего положения.

    Управление строится на разности текущей и целевой энергии системы.
    """

    def __init__(
        self, config: ControllerConfig, K: float, plant_config: PlantConfig
    ) -> None:
        super().__init__(config)
        self.name = "SwingUp"
        self._K = K
        self._plant_config = plant_config.copy()

    def get_control(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
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
        logger: Optional[Logger] = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:
        """SwingUp не обучается — управление задаётся аналитически."""
        raise NotImplementedError("SwingUp не требует обучения.")


class SwingUpAndBalance(Controller):
    """Композитный контроллер: раскачка (SwingUp) + балансировка (PID).

    Переключается между ``SwingUp`` (когда маятник далеко от вертикали)
    и ``PIDController`` (когда маятник близко к вертикали).
    """

    def __init__(
        self,
        config: ControllerConfig,
        swingup_controller: SwingUp,
        balance_controller: PIDController,
    ) -> None:
        super().__init__(config)
        self.name = "BEAST"
        self._swing_up_controller = swingup_controller
        self._balance_controller = balance_controller

    def get_control(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
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
        logger: Optional[Logger] = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:
        """Композитный контроллер не обучается напрямую."""
        raise NotImplementedError("SwingUpAndBalance не требует обучения.")

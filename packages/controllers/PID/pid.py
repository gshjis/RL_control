from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from loggers import Logger
from numpy.typing import NDArray

from packages.simulation.CO import (
    Controller,
    ControllerConfig,
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorConfig,
)


def terminate_condition(state: ObjectOfControl) -> bool:
    """Принимает объект управления, возвращает признак отклонения более 40°."""
    return abs(state.q[1] - np.pi) > np.radians(40)


class PIDController(Controller):
    """Принимает конфигурацию и коэффициенты, возвращает ПИД-силу."""

    def __init__(
        self, config: ControllerConfig, gains: np.ndarray | None = None
    ) -> None:
        """Принимает конфигурацию и необязательный вектор коэффициентов."""
        super().__init__(config)
        self.name = "PID"

        if gains is None:
            gains = np.array([10.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float64)
        self._Kp: float = float(gains[0])
        self._Ki: float = float(gains[1])
        self._Kd: float = float(gains[2])
        self._Kx: float = float(gains[3])
        self._Kdx: float = float(gains[4])

        self._integral: float = 0.0

    # ── Свойства ──────────────────────────────────────────────────────────

    @property
    def gains(self) -> NDArray[np.float64]:
        """Возвращает вектор коэффициентов ``[Kp, Ki, Kd, Kx, Kdx]``."""
        return np.array(
            [self._Kp, self._Ki, self._Kd, self._Kx, self._Kdx], dtype=np.float64
        )

    @gains.setter
    def gains(self, value: list[float] | NDArray[np.float64]) -> None:
        """Принимает вектор и устанавливает пять коэффициентов регулятора."""
        self._Kp, self._Ki, self._Kd, self._Kx, self._Kdx = map(float, value)

    # ── Закон управления ──────────────────────────────────────────────────

    def get_control(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
        """Принимает состояние и цель, возвращает ненасыщенную ПИД-силу."""
        error = target_state - s_clean

        self._integral += error[1] * self._dt

        F = (
            self._Kp * error[1]
            + self._Ki * self._integral
            + self._Kd * error[4]
            + self._Kx * error[0]
            + self._Kdx * error[3]
        )
        return F

    def reset_angel_integral(self) -> None:
        """Сбросить интегральную составляющую (для переключения уставки)."""
        self._integral = 0.0

    # ── Сброс ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Сбрасывает интегратор и состояние базового контроллера."""
        super().reset()
        self._integral = 0.0

    # ── Обучение ──────────────────────────────────────────────────────────

    def train(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        target_state: np.ndarray | Callable,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
        episode_max_time: float = 150.0,
        logger: Logger | None = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:
        """Принимает параметры среды и запускает выбранный оптимизатор."""
        method_options = method_options or {}
        optimizer = method_options.get("optimizer")
        if optimizer is None:
            raise ValueError(
                "PIDController.train() требует method_options['optimizer']"
            )
        print("Нихуя тут не происходит. Отъебись.")

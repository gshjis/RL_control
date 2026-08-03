from __future__ import annotations

from typing import Any, Callable, Optional
import cost_functions as cost_f
from loggers import Logger
import numpy as np
from numpy.typing import NDArray
from packages.simulation.CO import (
    Controller,
    ControllerConfig,
    ObjectOfControl,
    PlantConfig,
    SensorBlock,
    SensorConfig,
    NoiseForce
)
from packages.simulation.ENV.env import PendulumEnv


def terminate_condition(state: ObjectOfControl) -> bool:
    """
    Условие досрочного завершения эпизода.

    Текущая реализация всегда возвращает ``False`` (эпизод не прерывается).

    Parameters
    ----------
    state : ObjectOfControl
        Текущее состояние объекта управления.

    Returns
    -------
    bool
        ``True`` — эпизод следует прервать.

    Notes
    -----
    **TODO**: В текущей реализации функция всегда возвращает ``False``.
    Для активации проверки угла раскомментировать:

    .. code:: python

        if abs(state.q[1] - np.pi) > np.radians(40):
            return True
        return False
    """
    if abs(state.q[1] - np.pi) > np.radians(40):
        return True
    return False


class PIDController(Controller):
    """
    ПИД-регулятор с демпфированием по положению и скорости тележки.

    Закон управления:

    .. math::

        u = K_p \\cdot e_\\theta
          + K_i \\cdot \\int e_\\theta \\, dt
          + K_d \\cdot \\dot{e}_\\theta
          + K_x \\cdot e_x
          + K_{dx} \\cdot \\dot{e}_x

    где :math:`e = target\_state - s\_clean`.

    Parameters
    ----------
    config : ControllerConfig
        Конфигурация регулятора (такт, макс. сила, фильтры).
    gains : np.ndarray | None
        Вектор коэффициентов ``[Kp, Ki, Kd, Kx, Kdx]``.
        Если ``None`` — используется ``[10, 1, 2, 1, 2]``.

    Notes
    -----
    Optimization potential:
        - Вычисление ``error[1] * self._dt`` на каждом шаге
          можно предварительно умножить: ``self._Ki_dt = Ki * dt``.
        - ``get_action`` вызывается в горячем цикле ``clock_cycle``;
          стоит минимизировать количество индексаций массивов.
    """

    def __init__(self, config: ControllerConfig, gains: np.ndarray | None = None) -> None:
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
        """
        Текущий вектор коэффициентов ``[Kp, Ki, Kd, Kx, Kdx]``.

        Returns
        -------
        NDArray[np.float64]
            Массив коэффициентов (5,).
        """
        return np.array([self._Kp, self._Ki, self._Kd, self._Kx, self._Kdx], dtype=np.float64)

    @gains.setter
    def gains(self, value: list[float] | NDArray[np.float64]) -> None:
        """
        Установить все коэффициенты одним массивом.

        Parameters
        ----------
        value : list[float] | NDArray[np.float64]
            Вектор ``[Kp, Ki, Kd, Kx, Kdx]``.
        """
        self._Kp, self._Ki, self._Kd, self._Kx, self._Kdx = map(float, value)

    # ── Закон управления ──────────────────────────────────────────────────

    def get_control(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
        """
        Вычислить управляющее воздействие по ПИД-закону.

        Parameters
        ----------
        s_clean : np.ndarray
            Отфильтрованный вектор состояния ``(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)``.
        target_state : np.ndarray
            Целевой вектор состояния.

        Returns
        -------
        float
            Управляющая сила (Н) **до** насыщения.
        """
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
        """
        Сбросить PID-регулятор и базовый контроллер.

        Вызывать в начале каждого эпизода.
        """
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
        logger: Optional[Logger] = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Запустить оптимизацию коэффициентов ПИД-регулятора.

        Создаёт единую Gym-среду ``PendulumEnv`` с PID-контроллером
        и вызывает ``optimizer.optimize()`` для подбора коэффициентов.

        Parameters
        ----------
        plant_config : PlantConfig
            Конфигурация физической модели.
        sensor_config : SensorConfig
            Конфигурация датчиков.
        noise : NoiseForce
            Параметры внешнего возмущения.
        target_state : np.ndarray | Callable
            Целевое состояние или функция его генерации.
        terminate_condition : Callable | None
            Условие досрочного завершения эпизода.
        episode_max_time : float
            Максимальная длительность эпизода (с).
        logger : Logger | None
            Логгер для визуализации/логирования.
        method_options : dict | None
            Дополнительные параметры. Обязательный ключ ``optimizer`` —
            объект оптимизатора с методом ``optimize``.
        """
        method_options = method_options or {}
        optimizer = method_options.get("optimizer")
        if optimizer is None:
            raise ValueError(
                "PIDController.train() требует method_options['optimizer']"
            )

        env = PendulumEnv(
            plant_config=plant_config,
            sensor_config=sensor_config,
            controller=self,
            noise_force=noise,
            target_state=target_state,
            max_force=self._max_force,
        )
        result = optimizer.optimize(
            self,
            env,
            noise,
            target_state,
            terminate_condition,
            episode_max_time,
            logger,
        )
        print(result)

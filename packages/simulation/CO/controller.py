from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from loggers import Logger
from numpy.typing import NDArray

from packages.simulation.CO.datatypes import (
    ControllerConfig,
    NoiseForce,
    PlantConfig,
    SensorConfig,
)
from packages.simulation.CO.pendulum import ObjectOfControl
from packages.simulation.CO.signal_processing import Differentiator, SignalFilter


class Controller(ABC):
    """
    Абстрактное устройство управления (регулятор).

    Реализует **шаблонный метод** (Template Method) :meth:`compute_control`,
    который задаёт жёсткий конвейер обработки сигнала,
    общий для любых законов управления:

    1. Приём ``measured_state``
    2. Вычисление скоростей (дифференцирование) при отсутствии датчиков
    3. Фильтрация всего вектора состояния (ФНЧ)
    4. Вызов абстрактного :meth:`get_action` (закон управления)
    5. Клиппинг силы в диапазон ``[-max_force, +max_force]``
    6. Модель инерционности мотора (опционально)
    7. Сохранение и возврат ``F_ideal``

    Parameters
    ----------
    config : ControllerConfig
        Типизированная конфигурация регулятора (такт, макс. сила, фильтры).

    Notes
    -----
    Чтобы добавить новый закон управления, унаследуйтесь от ``Controller``
    и реализуйте :meth:`get_action`:

    >>> class MyController(Controller):
    ...     def get_action(self, s_clean, target_state) -> float:
    ...         return -s_clean[1]  # пропорционально углу

    Optimization potential:
        - ``np.concat`` в ``compute_control`` создаёт новый массив на
          каждом такте; можно переиспользовать предварительно выделенный
          буфер размером 6.
        - ``SignalFilter`` и ``Differentiator`` можно объединить в один
          проход (сейчас два последовательных EMA).
    """

    def __init__(self, config: ControllerConfig) -> None:
        """
        Parameters
        ----------
        config : ControllerConfig
            Конфигурация регулятора. Поддерживается только типизированный
            датакласс (словари больше не принимаются).

        Raises
        ------
        AttributeError
            Если в ``config`` отсутствуют необходимые поля.
        """
        # ── Извлечение параметров ─────────────────────────────────────

        _dt = config.dt
        _max_force = config.max_force
        _has_vel = config.has_velocity_sensors
        _diff_cutoff = config.differentiator_cutoff_hz
        _filter_cutoff = config.filter_cutoff_hz

        self.name: str
        self._dt: float = _dt
        self._max_force: float = _max_force
        self._has_velocity_sensors: bool = _has_vel

        # Компоненты обработки сигналов
        self._differentiator = Differentiator(
            dt=self._dt,
            cutoff_hz=_diff_cutoff if _diff_cutoff is not None else None,
        )
        self._signal_filter = SignalFilter(
            cutoff_hz=_filter_cutoff, dt=self._dt
        )

        # Память
        self._last_control_action: float = 0.0

    @property
    def last_control_action(self) -> float:
        """Последнее вычисленное значение силы (Н)."""
        return self._last_control_action

    @property
    def differentiator(self) -> Differentiator:
        """Блок численного дифференцирования скоростей."""
        return self._differentiator

    @property
    def signal_filter(self) -> SignalFilter:
        """Блок ФНЧ для сглаживания измерений."""
        return self._signal_filter

    @property
    def dt(self) -> float:
        """Такт управления (с)."""
        return self._dt

    def action(
        self, measured_state: np.ndarray, target_state: np.ndarray
    ) -> float:
        """
        Основной рабочий метод (Template Method).

        Pipeline:
        1. Извлечь координаты из ``measured_state``.
        2. Если датчиков скоростей нет — вычислить скорости через
           ``differentiator.calculate_velocity()``.
        3. Собрать полный вектор ``(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)`` и пропустить
           через ``signal_filter.filter_signal()``.
        4. Вызвать абстрактный :meth:`get_action(s_clean, target_state)`.
        5. Ограничить силу диапазоном ``[-max_force, +max_force]``.
        6. Сохранить в ``last_control_action`` и вернуть.

        Parameters
        ----------
        measured_state : np.ndarray
            Зашумлённый и/или квантованный вектор состояния с датчиков
            ``(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)``.
        target_state : np.ndarray
            Целевой вектор состояния ``(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)``.

        Returns
        -------
        float
            Реальная управляющая сила, приложенная к тележке (Н).

        Examples
        --------
        >>> from datatypes import ControllerConfig
        >>> cfg = ControllerConfig(dt=0.005, max_force=30.0)
        >>> ctrl = MyController(cfg)  # doctest: +SKIP
        >>> meas = np.array([0.0, 3.14, 0.0, 0.0, 0.0, 0.0])
        >>> target = np.zeros(6)
        >>> force = ctrl.compute_control(meas, target)  # doctest: +SKIP
        """


        # ── 1. Скорости ────────────────────────────────────────────────
        if self._has_velocity_sensors:
            velocities = measured_state[3:]
        else:
            velocities = self._differentiator.calculate_velocity(
                measured_state[:3]
            )

        # ── 2. Фильтрация ──────────────────────────────────────────────
        full = np.concat([measured_state[:3], velocities])
        s_clean = self._signal_filter.filter_signal(full)

        # ── 3. Закон управления (абстрактный) ──────────────────────────
        F_raw = self.get_control(s_clean, target_state)
            # ── 4. Насыщение (clipping) ────────────────────────────────────
        max_f = self._max_force
        if F_raw > max_f:
            F_clipped = max_f
        elif F_raw < -max_f:
            F_clipped = -max_f
        else:
            F_clipped = F_raw

        # ── 5. Сохранение и возврат ────────────────────────────────────
        self._last_control_action = float(F_clipped)
        return self._last_control_action

    @abstractmethod
    def get_control(
        self, s_clean: np.ndarray, target_state: np.ndarray
    ) -> float: 
        """
        Абстрактный метод вычисления управляющего воздействия.

        Переопределяется в классах-наследниках для реализации конкретного
        закона управления (ПИД, LQR, нейросеть и т.д.).

        Parameters
        ----------
        s_clean : np.ndarray
            Отфильтрованный вектор состояния
            ``(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)``.
        target_state : np.ndarray
            Целевой вектор состояния.

        Returns
        -------
        float
            Идеальная сила (Н) **до** насыщения.

        Notes
        -----
        Допускается возвращать значение за пределами ``[-max_force, +max_force]`` —
        ограничение будет применено в ``compute_control``.
        """
        ...

    @abstractmethod
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
        """
        Абстрактный метод обучения контроллера.

        Единый интерфейс для всех алгоритмов управления. Специфичные
        для конкретного алгоритма настройки передаются через
        ``method_options`` (например, ``{"optimizer": ...}`` для PID,
        ``{"epochs": ...}`` для PPO).

        Parameters
        ----------
        plant_config : PlantConfig
            Конфигурация физической модели.
        sensor_config : SensorConfig
            Конфигурация датчиков.
        noise : NoiseForce
            Параметры внешнего возмущения.
        target_state : np.ndarray
            Целевой вектор состояния.
        terminate_condition : Callable | None
            Условие досрочного завершения эпизода.
        episode_max_time : float
            Максимальная длительность эпизода (с).
        logger : Logger | None
            Опциональный логгер.
        method_options : dict | None
            Специфичные для алгоритма настройки.
        """
        ...

    def reset(self) -> None:
        """
        Сбросить внутреннюю память фильтра и дифференциатора.

        Вызывать в начале каждого нового эпизода, чтобы переходные
        процессы предыдущего запуска не влияли на старт.

        Examples
        --------
        >>> ctrl.reset()
        """
        self._differentiator.reset()
        self._signal_filter.reset()
        self._last_control_action = 0.0


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from packages.simulation.CO.datatypes import (
    ControllerConfig,
    NoiseForce,
)
from packages.simulation.CO.pendulum import ObjectOfControl
from packages.simulation.CO.sensor import SensorBlock


class Differentiator:
    """
    Блок численного дифференцирования с фильтрацией.

    Вычисляет вектор скорости по последовательным измерениям координат
    методом конечных разностей назад (backward difference).
    Результат дополнительно сглаживается ФНЧ первого порядка (EMA)
    для подавления шума квантования энкодеров.

    Parameters
    ----------
    dt : float
        Период дискретизации (с).
    cutoff_hz : float | None
        Частота среза ФНЧ для сглаживания скорости (Гц).
        Если ``None`` — фильтрация отключена (сырая производная).

    Notes
    -----
    Коэффициент сглаживания :math:`\\alpha = dt / (\\tau + dt)`,
    где :math:`\\tau = 1 / (2\\pi f_{cut})`.

    Optimization potential:
        - ``np.zeros_like()`` на первом вызове можно заменить на
          предварительно выделенный массив нулей (экономия аллокации).
        - ``positions.copy()`` вызывается дважды за шаг; можно
          переиспользовать буфер, если позволить модифицировать входной массив.
        - EMA эквивалентен ``np.lerp(self._filtered, raw_vel, self._alpha)``
          в NumPy ≥ 1.22 (одна операция вместо трёх).
    """

    def __init__(self, dt: float, cutoff_hz: float | None = None) -> None:
        self._dt = float(dt)
        self._prev_positions: NDArray[np.float64] | None = None
        self._filtered_velocity: NDArray[np.float64] | None = None

        # Коэффициент EMA-фильтра: alpha = dt / (tau + dt)
        if cutoff_hz is not None and cutoff_hz > 0.0:
            tau = 1.0 / (2.0 * np.pi * cutoff_hz)
            self._alpha = self._dt / (tau + self._dt)
        else:
            self._alpha = 1.0  # без фильтрации

    # ── Основной метод ────────────────────────────────────────────────────

    def calculate_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Вычислить скорость по текущему вектору координат.

        На первом вызове (нет предыдущего измерения) возвращает нулевой вектор.

        Parameters
        ----------
        positions : np.ndarray
            Координаты ``(x, θ₁, θ₂)`` на текущем шаге (формат (3,)).

        Returns
        -------
        np.ndarray
            Скорости ``(ẋ, θ̇₁, θ̇₂)`` (формат (3,)).

        Examples
        --------
        >>> diff = Differentiator(dt=0.01, cutoff_hz=30.0)
        >>> vel = diff.calculate_velocity(np.array([0.0, np.pi, 0.0]))
        >>> vel.shape
        (3,)
        """
        if self._prev_positions is None:
            self._prev_positions = positions.copy()
            return np.zeros_like(positions)

        # Сырая производная (backward difference)
        raw_vel = (positions - self._prev_positions) / self._dt

        # EMA-сглаживание
        if self._filtered_velocity is None:
            self._filtered_velocity = raw_vel.copy()
        else:
            self._filtered_velocity = (
                (1.0 - self._alpha) * self._filtered_velocity
                + self._alpha * raw_vel
            )

        self._prev_positions = positions.copy()
        return self._filtered_velocity

    # ── Сброс ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Сбросить внутреннюю историю (вызывать перед каждым эпизодом).

        После сброса следующий вызов ``calculate_velocity`` вернёт нули.
        """
        self._prev_positions = None
        self._filtered_velocity = None


class SignalFilter:
    """
    Блок экспоненциального сглаживания (ФНЧ первого порядка).

    Реализует фильтр :math:`y_k = (1-\\alpha)\\cdot y_{k-1} + \\alpha\\cdot u_k`
    с коэффициентом :math:`\\alpha = dt / (\\tau + dt)`, где
    :math:`\\tau = 1 / (2\\pi f_{cut})`.

    Parameters
    ----------
    cutoff_hz : float
        Частота среза фильтра (Гц). Должна быть > 0.
    dt : float
        Период дискретизации (с).

    Notes
    -----
    Optimization potential:
        - Аналогично ``Differentiator``: ``np.lerp`` может заменить две
          операции умножения.
        - При больших ``pool_size`` в ``SensorBlock`` фильтр может быть
          избыточен — шум уже усреднён пулом предвычисленных значений.
    """

    def __init__(self, cutoff_hz: float, dt: float) -> None:
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        self._alpha: float = dt / (tau + dt)
        self._filtered: NDArray[np.float64] | None = None

    # ── Основной метод ────────────────────────────────────────────────────

    def filter_signal(self, measurement: np.ndarray) -> np.ndarray:
        """
        Пропустить измерение через ФНЧ.

        На первом вызове (нет предыдущего значения) возвращает копию входа.

        Parameters
        ----------
        measurement : np.ndarray
            Входной зашумлённый вектор состояния (формат (6,)).

        Returns
        -------
        np.ndarray
            Сглаженный вектор состояния (формат (6,)).

        Examples
        --------
        >>> flt = SignalFilter(cutoff_hz=50.0, dt=0.005)
        >>> out = flt.filter_signal(np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0]))
        >>> out.shape
        (6,)
        """

        if self._filtered is None:
            self._filtered = measurement.copy()
        else:
            self._filtered = (
                (1.0 - self._alpha) * self._filtered
                + self._alpha * measurement
            )

        return self._filtered

    # ── Сброс ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Сбросить внутреннюю память фильтра.

        После сброса следующий вызов ``filter_signal`` вернёт копию входа.
        """
        self._filtered = None


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

    def set_motor_inertia(self, time_constant: float) -> None:
        """
        Установить модель инерционности двигателя.

        .. deprecated::
            Инерционность двигателя перенесена в объект управления
            (``PlantConfig.motor_time_constant``). Данный метод
            ничего не делает и сохранён только для обратной совместимости.

        Parameters
        ----------
        time_constant : float
            Игнорируется. Установите ``motor_time_constant`` в ``PlantConfig``.
        """
        import warnings
        warnings.warn(
            "Motor inertia is now a property of the plant (PlantConfig.motor_time_constant). "
            "set_motor_inertia() is deprecated and does nothing.",
            DeprecationWarning,
            stacklevel=2,
        )

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


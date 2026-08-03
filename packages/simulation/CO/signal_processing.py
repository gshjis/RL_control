from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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

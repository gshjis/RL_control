"""Условия завершения (terminate) эпизода."""

from __future__ import annotations

import numpy as np

from packages.simulation.co.layout import IDX_X


def terminate_condition(state: np.ndarray, target: np.ndarray) -> bool:
    """Принимает телеметрию и цель, возвращает признак завершения эпизода.

    Эпизод считается завершённым, если тележка выехала за пределы
    допустимого диапазона или состояние стало нечисловым (NaN/inf).
    """
    del target

    if not np.all(np.isfinite(state)):
        return True

    cart_out_of_bounds = np.abs(state[IDX_X]) > 0.5

    return bool(cart_out_of_bounds)

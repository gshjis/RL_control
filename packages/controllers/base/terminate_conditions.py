"""Условия аварийного завершения эпизода."""

import numpy as np


def terminate_condition(state: np.ndarray, target: np.ndarray) -> bool:
    """Принимает состояние и цель, возвращает признак аварийного завершения."""
    del target
    return abs(float(state[0])) > 0.5 or not np.all(np.isfinite(state))

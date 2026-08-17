"""Условия естественного усечения эпизода."""

import numpy as np


def truncated_condition(state: np.ndarray, target: np.ndarray) -> bool:
    """Принимает состояние и цель, возвращает признак усечения эпизода."""
    del state, target
    return False

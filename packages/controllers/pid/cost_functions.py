import numpy as np


def J(targ: np.ndarray, real: np.ndarray) -> float:
    """Принимает цель и состояние, возвращает сумму квадратов ошибки."""
    diff = targ - real
    return np.dot(diff, diff)

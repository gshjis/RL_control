"""Условия завершения (terminate) эпизода."""

from __future__ import annotations

import numpy as np

# TODO(experiment): импорты ниже нужны версии условия с проверками; пока
# terminate отключён, они не используются. Вернуть вместе с телом функции.
from packages.simulation.co.layout import IDX_X  # noqa: F401


def terminate_condition(state: np.ndarray, target: np.ndarray) -> bool:
    """Принимает телеметрию и цель, возвращает признак завершения эпизода.

    TODO(experiment): досрочное завершение временно отключено. Раньше эпизод
    рвался при выезде тележки за |x| > 0.5 (и при NaN/inf), из-за чего при
    b_c=0 тележка уезжала за ~1.2 с, и политика не успевала раскачать
    маятник. Вернуть проверки, когда политика научится удерживать тележку
    (и координата x появится в наблюдении):

        if not np.all(np.isfinite(state)):
            return True
        return bool(np.abs(state[IDX_X]) > 0.5)
    """
    return False

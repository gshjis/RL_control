"""Проверки функций награды и условий завершения эпизода."""

from __future__ import annotations

import numpy as np
import pytest

from packages.controllers.base import (
    make_energy_reward,
    terminate_condition,
    truncated_condition,
)
from packages.simulation.co.datatypes import PlantConfig
from packages.simulation.co.layout import (
    IDX_COS_T1,
    IDX_DTH1,
    IDX_X,
    TARGET_DIM,
    TELEMETRY_DIM,
)

# Параметры, для которых U_max = 2 * m1 * |g| * L1 = 2.0 Дж.
PLANT = PlantConfig(m1=0.5, l1=0.4, g=-10.0)
U_MAX = 2.0


def _telemetry(cos_theta1: float = 0.0, dtheta1: float = 0.0) -> np.ndarray:
    """Возвращает нулевую телеметрию с заданным cos(theta1) и скоростью."""
    state = np.zeros(TELEMETRY_DIM, dtype=np.float64)
    state[IDX_COS_T1] = cos_theta1
    state[IDX_DTH1] = dtheta1
    return state


def test_energy_reward_is_zero_at_upright_without_motion() -> None:
    """В верхней точке без кинетической энергии reward = 0 (это максимум)."""
    reward = make_energy_reward(PLANT)

    assert reward(_telemetry(cos_theta1=-1.0), np.ones(TARGET_DIM)) == pytest.approx(
        0.0
    )


def test_energy_reward_is_minus_u_max_at_the_bottom() -> None:
    """В нижней точке без движения reward = -U_max (это минимум)."""
    reward = make_energy_reward(PLANT)

    assert reward(_telemetry(cos_theta1=1.0), np.ones(TARGET_DIM)) == pytest.approx(
        -U_MAX
    )


def test_energy_reward_penalises_kinetic_energy() -> None:
    """Кинетическая энергия отдаляет систему от верхней точки и ухудшает reward."""
    reward = make_energy_reward(PLANT)
    target = np.ones(TARGET_DIM)

    stationary = reward(_telemetry(cos_theta1=-1.0, dtheta1=0.0), target)
    moving = reward(_telemetry(cos_theta1=-1.0, dtheta1=2.0), target)

    assert moving < stationary


def test_terminate_condition_detects_cart_limit() -> None:
    state = np.zeros(TELEMETRY_DIM, dtype=np.float64)
    state[IDX_X] = 0.51

    assert terminate_condition(state, state)


def test_terminate_condition_detects_non_finite_state() -> None:
    state = np.zeros(TELEMETRY_DIM, dtype=np.float64)
    state[IDX_COS_T1] = np.nan

    assert terminate_condition(state, state)


def test_terminate_condition_accepts_valid_state() -> None:
    state = np.zeros(TELEMETRY_DIM, dtype=np.float64)

    assert not terminate_condition(state, state)


def test_truncated_condition_is_false() -> None:
    state = np.zeros(TELEMETRY_DIM, dtype=np.float64)

    assert not truncated_condition(state, state)

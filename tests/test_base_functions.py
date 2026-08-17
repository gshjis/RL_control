import numpy as np

from packages.controllers.base import (
    make_energy_reward,
    terminate_condition,
    truncated_condition,
)
from packages.simulation.co.datatypes import PlantConfig


def test_energy_reward_factory_uses_plant_parameters() -> None:
    config = PlantConfig(m1=0.5, l1=0.4, g=-10.0)
    reward = make_energy_reward(config)

    state = np.zeros(8, dtype=np.float64)
    state[1] = 1.0
    state[6] = 2.0

    assert reward(state, np.ones(8)) > 0.0


def test_energy_reward_is_zero_at_lower_stationary_position() -> None:
    config = PlantConfig(m1=0.5, l1=0.4, g=-10.0)
    reward = make_energy_reward(config)
    state = np.zeros(8, dtype=np.float64)
    state[1] = 1.0

    assert reward(state, state) == 0.0


def test_terminate_condition_detects_cart_limit() -> None:
    state = np.zeros(8, dtype=np.float64)
    state[0] = 0.51

    assert terminate_condition(state, state)


def test_terminate_condition_detects_non_finite_state() -> None:
    state = np.zeros(8, dtype=np.float64)
    state[2] = np.nan

    assert terminate_condition(state, state)


def test_terminate_condition_accepts_valid_state() -> None:
    state = np.zeros(8, dtype=np.float64)

    assert not terminate_condition(state, state)


def test_truncated_condition_is_false() -> None:
    state = np.zeros(8, dtype=np.float64)

    assert not truncated_condition(state, state)

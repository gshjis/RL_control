"""Проверки единого описания эксперимента и именованных стратегий."""

from __future__ import annotations

import json

import numpy as np
import pytest

from packages.controllers.base import (
    build_reward,
    build_terminate,
    build_truncate,
)
from packages.experiment import build_target, upright_target
from packages.presets import default


def test_default_preset_to_dict_is_json_serializable() -> None:
    payload = default().to_dict()

    json.dumps(payload)  # не должен бросать

    assert payload["name"] == "ppo_upright"
    assert payload["reward"] == "energy"
    assert payload["terminate"] == "cart_limit"
    assert payload["truncate"] == "steps"
    assert payload["plant"]["M"] == 1.0
    assert payload["ppo"]["gamma"] == 0.999


def test_to_dict_is_deterministic() -> None:
    """Сериализация не должна зависеть от разовых выборок шума/углов."""
    assert default().to_dict() == default().to_dict()


def test_energy_reward_is_zero_at_upright_equilibrium() -> None:
    reward = build_reward("energy", default().plant)
    state = np.zeros(8, dtype=np.float64)
    state[1] = -1.0  # cos(theta = pi) — верхняя точка

    assert reward(state, upright_target(0.0)) == pytest.approx(0.0)


def test_energy_potential_reward_is_worse_at_the_bottom() -> None:
    reward = build_reward("energy_potential", default().plant)

    upright = np.zeros(8, dtype=np.float64)
    upright[1] = -1.0
    bottom = np.zeros(8, dtype=np.float64)
    bottom[1] = 1.0

    assert reward(upright, upright_target(0.0)) > reward(bottom, upright_target(0.0))


def test_unknown_strategy_error_lists_available_names() -> None:
    with pytest.raises(ValueError, match="energy"):
        build_reward("nope", default().plant)
    with pytest.raises(ValueError, match="cart_limit"):
        build_terminate("nope")
    with pytest.raises(ValueError, match="steps"):
        build_truncate("nope")
    with pytest.raises(ValueError, match="upright"):
        build_target("nope")


def test_with_plant_returns_independent_experiment() -> None:
    base = default()
    variant = base.with_plant(p_upright=0.05)

    assert variant.plant.p_upright == 0.05
    assert base.plant.p_upright != 0.05
    assert variant.reward == base.reward
    assert variant.ppo is base.ppo  # замороженная конфигурация общая


def test_with_plant_rejects_unknown_field() -> None:
    with pytest.raises(AttributeError):
        default().with_plant(nonexistent=1.0)


def test_upright_target_layout() -> None:
    np.testing.assert_array_equal(
        upright_target(0.0), np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    )


def test_default_strategies_are_buildable() -> None:
    experiment = default()

    assert callable(experiment.make_reward())
    assert callable(experiment.make_terminate())
    assert callable(experiment.make_truncate())
    assert callable(experiment.make_target())

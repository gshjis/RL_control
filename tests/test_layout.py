"""Проверки разметки векторов наблюдения, телеметрии и чистого состояния."""

from __future__ import annotations

import numpy as np
import pytest

from packages.simulation.co.layout import (
    CLEAN_STATE_DIM,
    IDX_COS_T1,
    IDX_DTH1,
    IDX_SIN_T1,
    OBS_DIM,
    TARGET_DIM,
    TELEMETRY_DIM,
    TELEMETRY_LABELS,
    split_observation,
    unpack_telemetry,
)


def test_dimensions_are_consistent() -> None:
    assert OBS_DIM == TELEMETRY_DIM + TARGET_DIM
    assert TELEMETRY_DIM == 8
    assert TARGET_DIM == 6
    assert len(TELEMETRY_LABELS) == TELEMETRY_DIM


def test_unpack_telemetry_positions_components() -> None:
    values = np.arange(TELEMETRY_DIM, dtype=float)

    telemetry = unpack_telemetry(values)

    assert telemetry.x == values[0]
    assert telemetry.cos_t1 == values[IDX_COS_T1]
    assert telemetry.dtheta1 == values[IDX_DTH1]


def test_unpack_telemetry_explains_angles() -> None:
    state = np.zeros(TELEMETRY_DIM, dtype=np.float64)
    state[IDX_COS_T1] = np.cos(0.5)
    state[IDX_SIN_T1] = np.sin(0.5)

    assert unpack_telemetry(state).theta1 == pytest.approx(0.5)


def test_split_observation_returns_telemetry_and_target() -> None:
    observation = np.arange(OBS_DIM, dtype=float)

    telemetry, target = split_observation(observation)

    assert telemetry.shape == (TELEMETRY_DIM,)
    assert target.shape == (TARGET_DIM,)
    np.testing.assert_array_equal(target, observation[TELEMETRY_DIM:])


def test_split_observation_rejects_wrong_size() -> None:
    with pytest.raises(ValueError, match=str(OBS_DIM)):
        split_observation(np.zeros(CLEAN_STATE_DIM))

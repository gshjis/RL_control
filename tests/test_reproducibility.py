"""Проверки воспроизводимости симуляции: сиды среды и C++-рандома.

Требует собранного C++-ядра (``make build``): без него импорт
``packages.simulation.co`` поднимает ImportError с подсказкой.
"""

from __future__ import annotations

import numpy as np

from packages.experiment import build_env
from packages.presets import default


def _rollout_fingerprint(seed: int, *, steps: int = 40) -> np.ndarray:
    """Прогоняет эпизод с фиксированными действиями и возвращает кадры."""
    experiment = default()
    env = build_env(experiment)

    observation, _ = env.reset(seed=seed)
    frames = [observation.copy()]

    # Одинаковая последовательность действий для обоих прогонов.
    action_rng = np.random.default_rng(7)
    for _ in range(steps):
        action = np.array([action_rng.uniform(-5.0, 5.0)], dtype=np.float64)
        observation, _reward, terminated, truncated, _info = env.step(action)
        frames.append(np.asarray(observation, dtype=np.float64).copy())
        if terminated or truncated:
            break

    env.close()
    return np.asarray(frames, dtype=np.float64)


def test_same_seed_gives_identical_trajectory() -> None:
    """Два прогона с одним seed совпадают побитово."""
    first = _rollout_fingerprint(1234)
    second = _rollout_fingerprint(1234)
    np.testing.assert_array_equal(first, second)


def test_different_seed_gives_different_trajectory() -> None:
    """Разные seed дают различающиеся траектории."""
    first = _rollout_fingerprint(1234)
    second = _rollout_fingerprint(4321)
    assert not np.allclose(first, second)


def test_constructor_seed_makes_first_reset_reproducible() -> None:
    """Фабрика сред (seed + rank) делает первый reset детерминированным."""
    experiment = default()
    env_a = build_env(experiment, seed=99)
    env_b = build_env(experiment, seed=99)

    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()

    np.testing.assert_array_equal(obs_a, obs_b)
    env_a.close()
    env_b.close()


def test_consecutive_resets_continue_the_stream() -> None:
    """Без seed повторный reset продолжает поток, а не копирует прошлый эпизод."""
    experiment = default()
    env = build_env(experiment, seed=5)

    first, _ = env.reset()
    second, _ = env.reset()

    assert not np.array_equal(first, second)
    env.close()

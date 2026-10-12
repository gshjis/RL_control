"""Тесты общего модуля прогонов (packages/rollout.py) без физики и C++."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from packages.rollout import RolloutResult, plot_rollout, rollout


class _StubEnv:
    """Минимальная среда: линейная динамика, нужна только для rollout()."""

    controller_dt = 0.01

    def __init__(self, *, terminate_at: int | None = None) -> None:
        self._step = 0
        self._terminate_at = terminate_at
        self.closed = False

    def reset(self, **_kwargs: object) -> tuple[np.ndarray, dict]:
        self._step = 0
        return np.zeros(14, dtype=np.float64), {}

    def step(self, action: np.ndarray):
        self._step += 1
        observation = np.zeros(14, dtype=np.float64)
        observation[1] = 1.0  # cos(theta1) — «поймали» вертикаль
        reward = float(action[0])
        terminated = self._terminate_at is not None and self._step >= self._terminate_at
        return observation, reward, terminated, False, {}

    def close(self) -> None:
        self.closed = True


class _StubController:
    """Контроллер с постоянной силой — достаточно для прогона."""

    def __init__(self, force: float) -> None:
        self._force = force

    def act(self, _observation: np.ndarray) -> np.ndarray:
        return np.array([self._force], dtype=np.float64)


def test_rollout_shapes_and_rewards() -> None:
    """Прогон собирает ряд нужной длины и суммирует награду."""
    env = _StubEnv()
    controller = _StubController(force=0.5)

    result = rollout(env, controller, seconds=0.1, runs=3)  # 10 шагов по 10 мс

    assert result.cos_t1.shape == (3, 10)
    assert result.times.shape == (10,)
    assert np.allclose(result.times, np.arange(10) * 0.01)
    # Награда суммируется по шагам эпизода: 10 × 0.5.
    assert np.allclose(result.rewards, 5.0)
    assert result.truncated_runs == 0
    assert env.closed


def test_rollout_keeps_last_value_after_early_termination() -> None:
    """При обрыве эпизода последнее значение дописывается до конца ряда."""
    env = _StubEnv(terminate_at=3)
    controller = _StubController(force=0.0)

    result = rollout(env, controller, seconds=0.1, runs=2)

    assert result.truncated_runs == 2
    # Шаг 0 (reset) + шаги 1..3 до обрыва; остальное — дописанное последнее.
    assert np.allclose(result.cos_t1[:, 3:], 1.0)
    # Награда только за фактически пройденные шаги.
    assert np.allclose(result.rewards, 0.0)


def test_rollout_sample_every_keeps_time_axis() -> None:
    """stride не сжимает ось времени: в ряд попадает каждая N-я выборка."""
    env = _StubEnv()
    controller = _StubController(force=1.0)

    result = rollout(env, controller, seconds=0.1, runs=1, sample_every=4)

    assert result.cos_t1.shape == (1, 3)  # шаги 0, 4, 8
    assert np.allclose(result.times, [0.0, 0.04, 0.08])


def test_rollout_rejects_invalid_arguments() -> None:
    """Некорректные параметры — ValueError, а не тихая работа."""
    env = _StubEnv()
    controller = _StubController(force=0.0)

    try:
        rollout(env, controller, seconds=1.0, runs=0)
        raise AssertionError("expected ValueError for runs=0")
    except ValueError:
        pass

    try:
        rollout(env, controller, seconds=0.0, runs=1)
        raise AssertionError("expected ValueError for seconds=0")
    except ValueError:
        pass


def test_summary_is_json_serialisable() -> None:
    """Сводка содержит только простые типы (попадает в metrics.json)."""
    import json

    result = RolloutResult(
        times=np.array([0.0, 0.01]),
        cos_t1=np.array([[1.0, 0.9], [0.8, 0.7]]),
        rewards=np.array([-1.0, -2.0]),
        truncated_runs=1,
    )

    summary = result.summary()
    assert summary["runs"] == 2
    assert summary["reward_mean"] == -1.5
    assert summary["truncated_runs"] == 1
    json.dumps(summary)  # не должно бросать


def test_plot_rollout_writes_png(tmp_path: Path) -> None:
    """График сохраняется в указанный путь."""
    result = RolloutResult(
        times=np.array([0.0, 0.01]),
        cos_t1=np.array([[1.0, 0.9]]),
        rewards=np.array([-1.0]),
        truncated_runs=0,
    )

    path = plot_rollout(result, tmp_path / "sub" / "plot.png")

    assert path.exists()
    assert path.stat().st_size > 0

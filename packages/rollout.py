"""Общие прогоны политики (rollout) и построение графиков.

Один и тот же цикл «прогнать политику N раз и снять статистику по
cos(θ₁) и награде» раньше был продублирован в `plot_creator.py`,
`train_and_evaluate_4_agents.py` и удалённой функции GUI-скрипта.
Теперь все headless-инструменты используют этот модуль.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from packages.simulation.co.layout import IDX_COS_T1  # noqa: E402

if TYPE_CHECKING:
    from packages.controllers.base.base import Controller
    from packages.simulation.env.env import PendulumEnv


@dataclass(frozen=True)
class RolloutResult:
    """Результат прогонов политики: ряды cos(θ₁) и суммарные награды."""

    times: np.ndarray  # [n_time]
    cos_t1: np.ndarray  # [n_runs, n_time]
    rewards: np.ndarray  # [n_runs]
    truncated_runs: int  # число эпизодов, завершённых раньше срока

    @property
    def mean_cos(self) -> np.ndarray:
        """Возвращает среднее cos(θ₁) по прогонам в каждый момент времени."""
        return self.cos_t1.mean(axis=0)

    @property
    def var_cos(self) -> np.ndarray:
        """Возвращает дисперсию cos(θ₁) по прогонам в каждый момент времени."""
        return self.cos_t1.var(axis=0)

    def summary(self) -> dict[str, Any]:
        """Возвращает сводку для сохранения в JSON."""
        return {
            "runs": int(self.rewards.shape[0]),
            "n_time": int(self.times.shape[0]),
            "duration_s": float(self.times[-1]) if self.times.size else 0.0,
            "truncated_runs": int(self.truncated_runs),
            "reward_mean": float(self.rewards.mean()),
            "reward_std": float(self.rewards.std()),
            "reward_min": float(self.rewards.min()),
            "reward_max": float(self.rewards.max()),
            "cos_final_mean": float(self.mean_cos[-1]) if self.mean_cos.size else 0.0,
            "cos_final_std": (
                float(self.cos_t1[:, -1].std()) if self.cos_t1.shape[1] else 0.0
            ),
        }


def rollout(
    env: PendulumEnv,
    controller: Controller,
    *,
    seconds: float,
    runs: int,
    sample_every: int = 1,
) -> RolloutResult:
    """Принимает среду и контроллер, выполняет `runs` прогонов по `seconds`.

    Шагание выполняется на каждом такте контроллера, а в ряд попадает
    каждая `sample_every`-я выборка — так временная шкала остаётся честной.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if seconds <= 0.0:
        raise ValueError("seconds must be > 0")

    sim_dt = float(env.controller_dt)
    max_steps = int(math.ceil(seconds / sim_dt))
    step_stride = max(int(sample_every), 1)
    record_indices = list(range(0, max_steps, step_stride))
    n_records = len(record_indices)

    times = np.asarray([index * sim_dt for index in record_indices], dtype=float)
    cos_series = np.zeros((runs, n_records), dtype=np.float64)
    rewards = np.zeros(runs, dtype=np.float64)
    truncated_runs = 0

    for run_index in range(runs):
        observation, _ = env.reset()
        last_cos = 0.0
        record_index = 0
        episode_reward = 0.0

        for step in range(max_steps):
            action = controller.act(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            last_cos = float(np.asarray(observation, dtype=np.float64)[IDX_COS_T1])

            if step == record_indices[record_index]:
                cos_series[run_index, record_index] = last_cos
                record_index += 1
                if record_index == n_records:
                    break

            if terminated or truncated:
                truncated_runs += 1
                break

        # Досрочное завершение эпизода: дописываем последнее значение.
        if record_index < n_records:
            cos_series[run_index, record_index:] = last_cos
        rewards[run_index] = episode_reward

    env.close()
    return RolloutResult(
        times=times,
        cos_t1=cos_series,
        rewards=rewards,
        truncated_runs=truncated_runs,
    )


def plot_rollout(
    result: RolloutResult,
    output_path: str | Path,
    *,
    title: str | None = None,
    spread_scale: float = 0.5,
) -> Path:
    """Принимает результат прогонов, сохраняет график, возвращает путь."""
    times = result.times
    mean_cos = result.mean_cos
    spread = spread_scale * np.sqrt(result.var_cos)

    figure, axis = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    axis.plot(times, mean_cos, linewidth=2.6, color="#1f77b4", label="mean cos(theta1)")
    axis.fill_between(
        times,
        mean_cos - spread,
        mean_cos + spread,
        color="#ff7f0e",
        alpha=0.18,
        label="dispersion (scaled spread)",
    )
    axis.set_title(title or f"cos(theta1): mean & dispersion over {result.rewards.size} runs")
    axis.set_xlabel("time, s")
    axis.set_ylabel("cos(theta1)")
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.set_ylim(-1.05, 1.05)
    axis.legend(frameon=False)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


__all__ = [
    "RolloutResult",
    "plot_rollout",
    "rollout",
]

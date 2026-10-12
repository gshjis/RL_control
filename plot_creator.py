"""Headless plot creator for the inverted pendulum.

Симулирует маятник ~10 секунд и сохраняет график cos(theta1).
Окна/GUI не открываются.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from packages.controllers.ppo import PPOController
from packages.experiment import ExperimentConfig, build_env
from packages.presets import default

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = (
    PROJECT_ROOT / "checkpoints" / "ppo" / "validation" / "model_1900000"
)

# Индекс cos(theta1) в векторе наблюдения (см. packages.simulation.co.layout).
COS_INDEX = 1


def load_controller(model_name: str | Path) -> PPOController:
    """Загружает обученную модель; бросает FileNotFoundError, если её нет."""
    controller_path_base = str(model_name)
    if (
        Path(f"{controller_path_base}.zip").exists()
        or Path(f"{controller_path_base}_model.zip").exists()
    ):
        print(f"Загружена модель: {controller_path_base}")
        return PPOController.load(controller_path_base)

    raise FileNotFoundError(
        f"Модель не найдена: {controller_path_base}.zip или *_model.zip"
    )


def simulate_and_save_cos_plot(
    *,
    model_name: str | Path = DEFAULT_MODEL_NAME,
    experiment: ExperimentConfig | None = None,
    max_real_time_s: float = 20.0,
    n_runs: int = 100,
    output_path: str | Path = PROJECT_ROOT
    / "plots"
    / "cos_theta1_mean_and_variance.png",
    render_every: int = 1,
) -> Path:
    """Делает n_runs прогонов, затем сохраняет среднюю и дисперсию cos(theta1).

    Останавливаемся по симуляционному времени (dt контроллера), как и в GUI.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if max_real_time_s <= 0.0:
        raise ValueError("max_real_time_s must be > 0")

    experiment = experiment if experiment is not None else default()
    env = build_env(experiment)
    controller = load_controller(model_name)

    sim_dt = float(experiment.controller.dt)
    max_steps = int(math.ceil(max_real_time_s / sim_dt))
    # Время считаем дискретно по t=step*dt,
    # а ряд записываем каждые render_every шагов.
    recorded_steps = list(range(0, max_steps, render_every))
    times = np.asarray([step * sim_dt for step in recorded_steps], dtype=float)

    # Собираем матрицу: [n_runs, n_time]
    all_cos = np.zeros((n_runs, len(recorded_steps)), dtype=float)

    for run_idx in range(n_runs):
        obs, _ = env.reset()
        for j, step in enumerate(recorded_steps):
            action = controller.action(obs)
            obs, _reward, terminated, truncated, _ = env.step(action)

            all_cos[run_idx, j] = float(np.asarray(obs, dtype=float)[COS_INDEX])

            if terminated or truncated:
                # Если досрочно — оставляем значения до конца как последние.
                if j + 1 < len(recorded_steps):
                    all_cos[run_idx, j + 1 :] = all_cos[run_idx, j]
                break

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_cos = all_cos.mean(axis=0)
    # «Дисперсия этого временного ряда»:
    var_cos = all_cos.var(axis=0)

    plt.figure(figsize=(12.5, 5.2), constrained_layout=True)
    ax = plt.gca()
    ax.plot(times, mean_cos, linewidth=2.6, color="#1f77b4", label="mean cos(theta1)")

    # Лёгкая визуальная «дисперсия»: рисуем полосу mean±0.5*sqrt(var)
    spread = np.sqrt(var_cos)
    ax.fill_between(
        times,
        mean_cos - 0.5 * spread,
        mean_cos + 0.5 * spread,
        color="#ff7f0e",
        alpha=0.18,
        label="dispersion (scaled spread)",
    )

    ax.set_title(f"cos(theta1): mean & dispersion over {n_runs} runs")
    ax.set_xlabel("time, s")
    ax.set_ylabel("cos(theta1)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(frameon=False)
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved plot: {output_path}")
    return output_path


def main() -> None:
    simulate_and_save_cos_plot()


if __name__ == "__main__":
    main()

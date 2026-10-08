"""Train 4 PPO agents with different upright-init probabilities and evaluate.

Что делает:
1) Обучает 4 агента PPO по 1_500_000 шагов.
2) Для каждого агента делает 100 rollout'ов по 5 секунд.
3) Сохраняет:
   - среднюю получаемую награду
   - cos(theta1) mean и дисперсию (var) во времени
   - все числовые данные в .npy и .json
   - график в PNG
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import (
    CONTROLLER_CONFIG,
    PLANT_CONFIG,
    SENSOR_CONFIG,
    ppo_config,
    target,
)
from packages.controllers.ppo import PPOController
from packages.simulation.env.env import PendulumEnv
from PPO_train import reward_f, terminate_condition, truncated_condition

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_ROOT = PROJECT_ROOT / "experiments" / "4_agents_upright_init"


def make_env(plant_config) -> PendulumEnv:
    return PendulumEnv(
        plant_config,
        SENSOR_CONFIG,
        reward_f,
        terminate_condition,
        CONTROLLER_CONFIG,
        target,
        truncated_condition,
        max_episode_steps=ppo_config.max_episode_steps,
    )


def eval_agent_cos_and_reward(
    *,
    controller: PPOController,
    plant_config,
    seconds: float,
    runs: int,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_dt = float(CONTROLLER_CONFIG.dt)
    max_steps = int(math.ceil(seconds / sim_dt))

    # В текущей разметке obs[1] соответствует cos(theta1)
    COS_INDEX = 1

    reward_sums = np.zeros(runs, dtype=float)
    cos_all = np.zeros((runs, max_steps), dtype=float)

    for run_idx in range(runs):
        env = make_env(plant_config)
        obs, _ = env.reset()
        ep_reward = 0.0

        for step_idx in range(max_steps):
            action = controller.action(obs)
            obs, reward, terminated, truncated, _info = env.step(action)

            ep_reward += float(reward)
            cos_all[run_idx, step_idx] = float(np.asarray(obs, dtype=float)[COS_INDEX])

            if terminated or truncated:
                last = cos_all[run_idx, step_idx]
                if step_idx + 1 < max_steps:
                    cos_all[run_idx, step_idx + 1 :] = last
                break

        reward_sums[run_idx] = ep_reward

    times = np.asarray([k * sim_dt for k in range(max_steps)], dtype=float)
    cos_mean = cos_all.mean(axis=0)
    cos_var = cos_all.var(axis=0)

    np.save(out_dir / "cos_times.npy", times)
    np.save(out_dir / "cos_all_runs.npy", cos_all)
    np.save(out_dir / "cos_mean.npy", cos_mean)
    np.save(out_dir / "cos_var.npy", cos_var)

    metrics = {
        "runs": runs,
        "seconds": seconds,
        "max_steps": max_steps,
        "reward_mean": float(reward_sums.mean()),
        "reward_std": float(reward_sums.std()),
        "reward_min": float(reward_sums.min()),
        "reward_max": float(reward_sums.max()),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # PNG
    cos_std = np.sqrt(cos_var)
    spread = 0.5 * cos_std

    fig, ax = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    ax.plot(times, cos_mean, linewidth=2.6, color="#1f77b4", label="mean cos(theta1)")
    ax.fill_between(
        times,
        cos_mean - spread,
        cos_mean + spread,
        color="#ff7f0e",
        alpha=0.18,
        label="dispersion (scaled)",
    )
    ax.set_title(f"cos(theta1): mean & dispersion over {runs} runs")
    ax.set_xlabel("time, s")
    ax.set_ylabel("cos(theta1)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(frameon=False)

    fig.savefig(
        out_dir / "cos_plot.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    return metrics


def main() -> None:
    agent_specs = [
        (0.00, "p0"),
        (0.01, "p1"),
        (0.05, "p5"),
        (0.10, "p10"),
    ]

    train_steps = 1_500_000
    eval_runs = 100
    eval_seconds = 5.0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for p_upright, tag in agent_specs:
        print(f"=== Training agent {tag} | p_upright={p_upright} ===")
        plant_cfg = PLANT_CONFIG.copy()
        plant_cfg.p_upright = float(p_upright)

        controller = PPOController(
            ppo_config=ppo_config,
            controller_config=CONTROLLER_CONFIG,
        )

        # В этом проекте train() использует EnvOrchestrator из PPOController.train.
        # Мы создаём EnvOrchestrator напрямую так же, как в PPO_train.py.
        from packages.simulation.env.env_orcestrator import EnvOrchestrator

        orchestrator = EnvOrchestrator(
            plant_cfg,
            SENSOR_CONFIG,
            reward_f,
            terminate_condition,
            10,
            CONTROLLER_CONFIG,
            truncated_condition,
            target,
            ppo_config.max_episode_steps,
        )

        controller.train(orchestrator)

        ckpt_dir = OUT_ROOT / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_base = ckpt_dir / f"ppo_agent_{tag}_steps_{train_steps}"
        controller.save(str(model_base))

        print(f"=== Evaluating agent {tag} ===")
        agent_out = OUT_ROOT / f"agent_{tag}"
        metrics = eval_agent_cos_and_reward(
            controller=controller,
            plant_config=plant_cfg,
            seconds=eval_seconds,
            runs=eval_runs,
            out_dir=agent_out,
        )

        summary[tag] = {"p_upright": p_upright, **metrics}

    with open(OUT_ROOT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. Summary: {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()

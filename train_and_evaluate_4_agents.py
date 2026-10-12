"""Train 4 PPO agents with different upright-init probabilities and evaluate.

Что делает:
1) Обучает 4 агента PPO (число шагов — из ``ExperimentConfig.ppo``).
2) Для каждого агента делает 100 rollout'ов по 5 секунд.
3) Сохраняет:
   - среднюю получаемую награду
   - cos(theta1) mean и дисперсию (var) во времени
   - все числовые данные в .npy и .json
   - график в PNG
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from packages.controllers.ppo import PPOController, PPOTrainer
from packages.experiment import ExperimentConfig, build_env, make_env_factory
from packages.presets import default
from packages.rollout import plot_rollout, rollout

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_ROOT = PROJECT_ROOT / "experiments" / "4_agents_upright_init"


def train_agent(experiment: ExperimentConfig) -> PPOController:
    """Обучает одного PPO-агента по описанию эксперимента."""
    trainer = PPOTrainer(
        ppo_config=experiment.ppo,
        controller_config=experiment.controller,
        env_factory=make_env_factory(experiment),
        n_envs=experiment.n_envs,
    )
    return trainer.fit()


def eval_agent_cos_and_reward(
    *,
    controller: PPOController,
    experiment: ExperimentConfig,
    seconds: float,
    runs: int,
    out_dir: Path,
) -> dict:
    """Оценивает агента: суммарная награда и статистика cos(theta1) во времени."""
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(experiment, max_episode_steps=experiment.ppo.max_episode_steps)
    result = rollout(env, controller, seconds=seconds, runs=runs)

    np.save(out_dir / "cos_times.npy", result.times)
    np.save(out_dir / "cos_all_runs.npy", result.cos_t1)
    np.save(out_dir / "cos_mean.npy", result.mean_cos)
    np.save(out_dir / "cos_var.npy", result.var_cos)

    metrics = {"seconds": seconds, **result.summary()}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_rollout(result, out_dir / "cos_plot.png")
    return metrics


def main() -> None:
    base = default()
    agent_specs = [
        (0.00, "p0"),
        (0.01, "p1"),
        (0.05, "p5"),
        (0.10, "p10"),
    ]

    train_steps = base.ppo.total_timesteps
    eval_runs = 100
    eval_seconds = 5.0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for p_upright, tag in agent_specs:
        print(f"=== Training agent {tag} | p_upright={p_upright} ===")
        experiment = base.with_plant(p_upright=float(p_upright))

        controller = train_agent(experiment)

        ckpt_dir = OUT_ROOT / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_base = ckpt_dir / f"ppo_agent_{tag}_steps_{train_steps}"
        controller.save(str(model_base))

        print(f"=== Evaluating agent {tag} ===")
        metrics = eval_agent_cos_and_reward(
            controller=controller,
            experiment=experiment,
            seconds=eval_seconds,
            runs=eval_runs,
            out_dir=OUT_ROOT / f"agent_{tag}",
        )

        summary[tag] = {"p_upright": p_upright, **metrics}

    with open(OUT_ROOT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. Summary: {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()

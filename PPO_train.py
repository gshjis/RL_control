"""Обучение PPO-агента на перевёрнутом одно-звенном маятнике."""

from __future__ import annotations

from packages.controllers.ppo import PPOController, PPOTrainer
from packages.experiment import ExperimentConfig, make_env_factory
from packages.presets import default

DEFAULT_CHECKPOINT = "checkpoints/ppo/pendl"


def train(
    experiment: ExperimentConfig,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> PPOController:
    """Обучает PPO-агента и сохраняет модель, нормализатор и конфигурацию."""
    trainer = PPOTrainer(
        ppo_config=experiment.ppo,
        controller_config=experiment.controller,
        env_factory=make_env_factory(experiment),
        n_envs=experiment.n_envs,
    )

    print("Обучение PPO...")
    print(f"  total_timesteps = {experiment.ppo.total_timesteps}")
    print(f"  n_envs = {experiment.n_envs}")
    print(f"  Инерция двигателя: τ = {experiment.plant.motor_time_constant} с")

    controller = trainer.fit()

    controller.save(checkpoint)
    print(f"Модель сохранена: {checkpoint}")
    print("Обучение PPO завершено!")
    return controller


def main() -> None:
    train(default())


if __name__ == "__main__":
    main()

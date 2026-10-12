"""Обучение PPO-агента на перевёрнутом одно-звенном маятнике."""

from __future__ import annotations

from packages.controllers.ppo import PPOController
from packages.experiment import ExperimentConfig, build_orchestrator
from packages.presets import default

DEFAULT_CHECKPOINT = "checkpoints/ppo/pendl"


def train(
    experiment: ExperimentConfig,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> PPOController:
    """Обучает PPO-агента и сохраняет модель, нормализатор и конфигурацию."""
    controller = PPOController(
        ppo_config=experiment.ppo,
        controller_config=experiment.controller,
    )

    print("Обучение PPO...")
    print(f"  total_timesteps = {experiment.ppo.total_timesteps}")
    print("  Пробел — сброс, C — мотор вкл/выкл, Q / ESC — выход")
    print(f"  Инерция двигателя: τ = {experiment.plant.motor_time_constant} с")

    controller.train(build_orchestrator(experiment))

    controller.save(checkpoint)
    print(f"Модель сохранена: {checkpoint}")
    print("Обучение PPO завершено!")
    return controller


def main() -> None:
    train(default())


if __name__ == "__main__":
    main()

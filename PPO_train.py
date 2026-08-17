"""
Основной скрипт: запуск симуляции перевёрнутого маятника с GUI.
"""

from __future__ import annotations

from configs import CONTROLLER_CONFIG, PLANT_CONFIG, SENSOR_CONFIG, ppo_config, target
from packages.controllers.base import (
    make_energy_reward,
    terminate_condition,
    truncated_condition,
)
from packages.controllers.ppo import PPOController
from packages.simulation.env import env_orcestrator

MODEL_NAME = "checkpoints/ppo/pendl"
reward_f = make_energy_reward(PLANT_CONFIG)


if __name__ == "__main__":
    ppo_controller = PPOController(
        ppo_config=ppo_config,
        controller_config=CONTROLLER_CONFIG,
    )

    orchestrator = env_orcestrator.EnvOrchestrator(
        PLANT_CONFIG,
        SENSOR_CONFIG,
        reward_f,
        terminate_condition,
        10,
        CONTROLLER_CONFIG,
        truncated_condition,
        target,
        ppo_config.max_episode_steps,
    )

    print("Обучение PPO...")
    print(f"  total_timesteps = {ppo_config.total_timesteps}")
    print("  Пробел — сброс, C — мотор вкл/выкл, Q / ESC — выход")
    print(f"  Инерция двигателя: τ = {PLANT_CONFIG.motor_time_constant} с")
    ppo_controller.train(orchestrator)

    # Сохранить модель, нормализатор и конфиги.
    ppo_controller.save(MODEL_NAME)
    print(f"Модель сохранена: {MODEL_NAME}")
    print("Обучение PPO завершено!")

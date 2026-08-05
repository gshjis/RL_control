"""
Основной скрипт: запуск симуляции перевёрнутого маятника с GUI.
"""

from __future__ import annotations

from configs import *
from packages.controllers.PPO import PPOController
from packages.simulation.ENV import env_orcestrator

if __name__ == "__main__":

    ppo_controller = PPOController(
        ppo_config=ppo_config,
        controller_config=CONTROLLER_CONFIG,
    )

    def terminate_condition(e) -> bool:
        return bool(abs(e[1]) > 0.2)

    def cost_f(error) -> float:

        return 1

    env_orcestrator = env_orcestrator.EnvOrchestrator(
        PLANT_CONFIG,
        SENSOR_CONFIG,
        cost_f,
        terminate_condition,
        8, 
        CONTROLLER_CONFIG,
        target)

    print("Обучение PPO...")
    print(f"  total_timesteps = {ppo_config.total_timesteps}")
    print("  Пробел — сброс, C — мотор вкл/выкл, Q / ESC — выход")
    print(f"  Инерция двигателя: τ = {PLANT_CONFIG.motor_time_constant} с")

    ppo_controller.train(
        env_orcestrator
    )
    print("Обучение PPO завершено!")
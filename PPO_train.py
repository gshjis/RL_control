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

    def terminate_condition(s, s_t) -> bool:
        return bool(abs(s[0] - s_t[0]) > 0.5 or abs(s[1] - s_t[0]) > 0.26)

    def cost_f(s, s_t)->float:
        diff = s_t - s
        return np.dot(diff, diff).item()

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
"""
Основной скрипт: запуск симуляции перевёрнутого маятника с GUI.
"""

from __future__ import annotations

from configs import *
from packages.controllers.PPO import PPOController
from packages.simulation.ENV import env_orcestrator

# Имя (базовое) для сохранения модели, нормализатора и конфигов.
MODEL_NAME = "checkpoints/ppo/2_pendl"

def terminate_condition(error) -> bool:
    return abs(error[0]) > 0.5

def cost_f(error) -> float:
    
    return np.exp(1/(error[1]+0.01))

def truncated_condition(error)->bool:
    return abs(error[0])<0.01 and abs(error[1])<0.1

if __name__ == "__main__":

    ppo_controller = PPOController(
        ppo_config=ppo_config,
        controller_config=CONTROLLER_CONFIG,
    )

    env_orcestrator = env_orcestrator.EnvOrchestrator(
        PLANT_CONFIG,
        SENSOR_CONFIG,
        cost_f,
        terminate_condition,
        10,
        CONTROLLER_CONFIG,
        truncated_condition,
        target
        )

    print("Обучение PPO...")
    print(f"  total_timesteps = {ppo_config.total_timesteps}")
    print("  Пробел — сброс, C — мотор вкл/выкл, Q / ESC — выход")
    print(f"  Инерция двигателя: τ = {PLANT_CONFIG.motor_time_constant} с")
    ppo_controller.train(
        env_orcestrator
    )

    # Сохранить модель, нормализатор и конфиги.
    ppo_controller.save(MODEL_NAME)
    print(f"Модель сохранена: {MODEL_NAME}")
    print("Обучение PPO завершено!")
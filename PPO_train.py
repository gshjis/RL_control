"""
Основной скрипт: запуск симуляции перевёрнутого маятника с GUI.
"""

from __future__ import annotations

from configs import *
from packages.controllers.PPO import PPOController
from packages.simulation.ENV import env_orcestrator

# Имя (базовое) для сохранения модели, нормализатора и конфигов.
MODEL_NAME = "checkpoints/ppo/2_pendl"
import numpy as np

def terminate_condition(error) -> bool:
    """Завершаем эпизод при падении маятника или выезде тележки."""
    # Падение маятника (угол > 60° от вертикали)
    if error[1] > 1.0:  # cos(θ) < 0.5 → угол > 60°
        return True
    # Выезд тележки за пределы
    if abs(error[0]) > 2.0:  # тележка уехала слишком далеко
        return True
    return False

def cost_f(error) -> float:
    """Награда: близость к вертикали + центр."""
    cos_error = error[1]   # ошибка по косинусу (чем меньше, тем лучше)
    x_error = error[0]     # ошибка по положению тележки
    
    # Квадратичный штраф (стабильный и понятный)
    reward = -cos_error**2 * 100.0
    
    # Бонус за удержание вертикали
    if abs(cos_error) < 0.05:
        reward += 50.0
    
        
    return reward

def truncated_condition(error) -> bool:
    """Успех — маятник в вертикали и тележка в центре."""
    cos_error = error[1]
    x_error = error[0]
    return abs(cos_error) < 0.02 and abs(x_error) < 0.02
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
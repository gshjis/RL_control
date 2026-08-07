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


def terminate_condition(state:np.ndarray, target:np.ndarray) -> bool:
    return abs(state[0]) > 0.5

def reward_f(state: np.ndarray, target: np.ndarray) -> float:
    # 1. Распаковка state (согласно твоему описанию)
    x = state[0]          # положение тележки
    cos_theta1 = state[1] # косинус угла 1-го звена (цель: -1)
    sin_theta1 = state[2] # синус угла 1-го звена (для направления)
    cos_theta2 = state[3] # косинус 2-го звена (сейчас 0)
    sin_theta2 = state[4] # синус 2-го звена (сейчас 0)
    dx = state[5]         # скорость тележки
    dtheta1 = state[6]    # угловая скорость 1-го звена
    dtheta2 = state[7]    # угловая скорость 2-го звена

    # 2. ОСНОВНАЯ НАГРАДА: за угол (подъем маятника)
    # cos = -1 (вверху) -> штраф 0. cos = 1 (внизу) -> штраф -4
    reward_angle = 1/((cos_theta1 + 1.0)**2 + 0.1)

    # 3. ШТРАФ ЗА ВЫЛЕТ ТЕЛЕЖКИ (чтобы не улетала за край)

    # 4. ШТРАФ ЗА СИЛУ (чтобы не дергалась без толку)
    # Коэффициент 0.001 — стандарт, чтобы большие силы не поощрялись

    # 5. СТАБИЛИЗАЦИЯ ВВЕРХУ (бонус за то, что поймал)
    # Если маятник почти встал (cos < -0.95) и скорость маленькая (|dtheta| < 1.0)
    # даем большую положительную награду, чтобы агент учился удерживать его там
    if cos_theta1 < -0.95 and abs(dtheta1) < 1.0:
        bonus_up = 10.0
    else:
        bonus_up = 0.0

    # 6. Итоговая награда
    reward = reward_angle + bonus_up

    return reward
    
def truncated_condition(state:np.ndarray, target:np.ndarray) -> bool:
    return False

if __name__ == "__main__":

    ppo_controller = PPOController(
        ppo_config=ppo_config,
        controller_config=CONTROLLER_CONFIG,
    )

    env_orcestrator = env_orcestrator.EnvOrchestrator(
        PLANT_CONFIG,
        SENSOR_CONFIG,
        reward_f,
        terminate_condition,
        5,
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

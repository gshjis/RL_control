"""
Основной скрипт: запуск симуляции перевёрнутого маятника с GUI.
"""

from __future__ import annotations

import numpy as np

from configs import CONTROLLER_CONFIG, PLANT_CONFIG, SENSOR_CONFIG, ppo_config, target
from packages.controllers.ppo import PPOController
from packages.simulation.env import env_orcestrator

MODEL_NAME = "checkpoints/ppo/pendl"


def terminate_condition(state: np.ndarray, target: np.ndarray) -> bool:
    """Принимает состояние и цель, возвращает признак аварийного завершения."""
    del target
    return abs(float(state[0])) > 0.5 or not np.all(np.isfinite(state))


def reward_f(state: np.ndarray, target: np.ndarray) -> float:
    """Принимает состояние и цель, возвращает энергию первого звена."""
    del target

    cos_theta1 = float(state[1])
    dtheta1 = float(state[6])

    # Energy of the first link only. The cart energy and the cart-pendulum
    # coupling are intentionally excluded. Parameters come from PLANT_CONFIG.
    m1 = float(PLANT_CONFIG.m1)
    L1 = float(PLANT_CONFIG.L1)
    J1 = float(PLANT_CONFIG.J1)
    gravity = abs(float(PLANT_CONFIG.g))
    inertia_about_pivot = J1 + m1 * L1 * L1
    kinetic_energy = 0.5 * inertia_about_pivot * dtheta1 * dtheta1
    potential_energy = m1 * gravity * L1 * (1.0 - cos_theta1)
    return float(kinetic_energy + potential_energy)


def truncated_condition(state: np.ndarray, target: np.ndarray) -> bool:
    """Принимает состояние и цель, возвращает признак усечения эпизода."""
    del state, target
    return False


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

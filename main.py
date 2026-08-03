"""
Основной скрипт: обучение PPO-контроллера и запуск GUI-симуляции.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from packages.controllers.PPO import PPOConfig, PPOController
from packages.simulation.CO import (
    ControllerConfig,
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorConfig,
)
from packages.simulation.GUI import PendulumViewer

# ═══════════════════════════════════════════════════════════════════════════
# Конфигурация физической модели
# ═══════════════════════════════════════════════════════════════════════════

PLANT_CONFIG = PlantConfig(
    M=1.0,
    m1=0.1,
    l1=0.3,
    m2=0.1,
    l2=0.3,
    g=-9.81,
    b_c=0.1,
    b_1=0.003,
    b_2=0.003,
    single_pendulum_mode=True,
    backslash_mode=False,
    init_q=np.array([0.0, np.pi, 0.0]),
    init_dq=np.array([0.0, 0.0, 0.0]),
    dt=0.0001,
)

SENSOR_CONFIG = SensorConfig(
    encoder_resolution_1=4096,
    encoder_resolution_2=4096,
    cart_sensor_resolution=0.0001,
    noise_std_q=(0.0005, 0.002, 0.002),
    noise_std_dq=(0.005, 0.01, 0.01),
)

CONTROLLER_CONFIG = ControllerConfig(
    dt=0.001,
    max_force=24.0,
    has_velocity_sensors=False,
    filter_cutoff_hz=50.0,
)

PPO_CFG = PPOConfig(
    total_timesteps=500_000,
    max_episode_steps=10000,
    dt_control=0.001,
)

NOISE = NoiseForce(mean=0.00, std=0.03)
TARGET = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])

MODEL_PATH = Path("models") / "ppo_controller.zip"


# ═══════════════════════════════════════════════════════════════════════════
# Точка входа
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Создание контроллера ────────────────────────────────────────────
    controller = PPOController(
        ppo_config=PPO_CFG,
        controller_config=CONTROLLER_CONFIG,
        plant_config=PLANT_CONFIG,
        sensor_config=SENSOR_CONFIG,
        noise=NOISE,
        target_state=TARGET,
    )

    # ── Обучение или загрузка ───────────────────────────────────────────
    if MODEL_PATH.exists():
        print(f"Загрузка предобученной модели: {MODEL_PATH}")
        controller = PPOController.from_pretrained(
            path=MODEL_PATH,
            ppo_config=PPO_CFG,
            controller_config=CONTROLLER_CONFIG,
            plant_config=PLANT_CONFIG,
            sensor_config=SENSOR_CONFIG,
            noise=NOISE,
            target_state=TARGET,
        )
    else:
        print("Обучение PPO-контроллера...")
        controller.train(
            plant_config=PLANT_CONFIG,
            sensor_config=SENSOR_CONFIG,
            noise=NOISE,
            target_state=TARGET,
        )
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        controller.save(MODEL_PATH)
        print(f"Модель сохранена: {MODEL_PATH}")

    # ── Запуск GUI ──────────────────────────────────────────────────────
    print("Запуск GUI-симуляции...")
    viewer = PendulumViewer(
        plant=ObjectOfControl(PLANT_CONFIG),
        sensor_config=SENSOR_CONFIG,
        noise=NOISE,
        target_state=TARGET,
        controller=controller,
    )
    viewer.use()

"""
Запуск GUI-симуляции перевёрнутого маятника.

Использование:
    poetry run python run_gui.py

Если найдена предобученная PPO-модель — используется она, иначе ручное
управление стрелками (←/→).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from configs import *
from configs import target
from packages.controllers.PPO import PPOConfig, PPOController
from packages.simulation.CO import (
    ControllerConfig,
    PlantConfig,
    SensorConfig,
)
from packages.simulation.ENV.env import PendulumEnv
from packages.simulation.GUI import PendulumViewer
from PPO_train import *

# ── Конфигурация ─────────────────────────────────────────────────────────

SENSOR_CONFIG = SensorConfig(
    encoder_resolution_1=4096,
    encoder_resolution_2=4096,
    cart_sensor_resolution=0.0001,
    noise_std_q=(0.0005, 0.002, 0.002),
    noise_std_dq=(0.005, 0.01, 0.01),
)

CONTROLLER_CONFIG = ControllerConfig(
    dt=0.01,
    max_force=24.0,
    has_velocity_sensors=False,
    filter_cutoff_hz=50.0,
)

TARGET = np.array([0.0, 1, 0.0, 0.0, 0.0, 0.0])

# Имя (базовое) сохранённой модели — должно совпадать с MODEL_NAME в PPO_train.py.

 
def main() -> None:
    # ── Среда ────────────────────────────────────────────────────────────
    env = PendulumEnv(
        PLANT_CONFIG,
        SENSOR_CONFIG,
        reward_f,
        terminate_condition,
        CONTROLLER_CONFIG,
        target,
        truncated_condition,
        )

    # ── Контроллер (если есть предобученная модель) ─────────────────────
    controller = None
    MODEL_NAME = "checkpoints/ppo/validation/model_400000"
    if Path(f"{MODEL_NAME}.zip").exists():
        controller = PPOController.load(MODEL_NAME)
        print(f"Загружена модель: {MODEL_NAME}")
    else:
        print(f"Модель {MODEL_NAME} не найдена — ручное управление (←/→).")

    viewer = PendulumViewer(env=env, controller=controller)
    viewer.use()


if __name__ == "__main__":
    main()
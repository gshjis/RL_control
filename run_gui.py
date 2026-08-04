"""
Запуск GUI-симуляции с предобученной PPO-моделью.

Использование:
    poetry run python run_gui.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from packages.controllers.PPO import PPOConfig, PPOController
from packages.simulation.CO import (
    ControllerConfig,
    NoiseForce,
    PlantConfig,
    SensorConfig,
)
from packages.simulation.ENV.env import PendulumEnv
from packages.simulation.GUI import PendulumViewer

# ── Конфигурация (та же, что в main.py) ────────────────────────────────
PLANT_CONFIG = PlantConfig(
    M=1.0,        
    m1=0.1,       
    l1=0.3,       
    m2=0.0,
    l2=0.0,
    g=-9.81,       
    b_c=0.01,     
    b_1=0.001,    
    b_2=0.001,
    single_pendulum_mode=True,
    backslash_mode=False,
    init_q=np.array([0.0, np.pi, 0.0]),
    init_dq=np.array([0.0, 0.0, 0.0]),
    dt=0.002,
    motor_time_constant=0.05
)

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

NOISE = NoiseForce(mean=0.00, std=0.03)
TARGET = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])

# Приоритет: final_model (с VecNormalize) → best_model
MODEL_PATH = "checkpoints/ppo/best/best_model.zip"
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "checkpoints/ppo/best/best_model.zip"


def main() -> None:
    # ── Загрузка предобученной модели ──────────────────────────────────
    controller = PPOController.from_pretrained(
        MODEL_PATH,
        ppo_config=PPOConfig(),
        controller_config=CONTROLLER_CONFIG,
    )
    print(f"Загружена модель: {MODEL_PATH}")
    print(f"  VecNormalize: {'есть' if controller._vec_normalize is not None else 'нет'}")

    # ── Создание среды и GUI ───────────────────────────────────────────
    env = PendulumEnv(
        PLANT_CONFIG,
        SENSOR_CONFIG,
        controller,
        NOISE,
        TARGET,
        max_force=CONTROLLER_CONFIG.max_force,
    )
    viewer = PendulumViewer(env=env)
    viewer.use()


if __name__ == "__main__":
    main()

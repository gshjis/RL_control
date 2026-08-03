"""
Основной скрипт: запуск симуляции перевёрнутого маятника с GUI.
"""

from __future__ import annotations

import numpy as np

from packages.controllers.PID import PIDController
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
    m2=0.0,
    l2=0.0,
    g=-9.81,
    b_c=0.1,
    b_1=0.003,
    b_2=0.003,
    single_pendulum_mode=True,
    backslash_mode=False,
    init_q=np.array([0.0, np.pi, 0.0]),
    init_dq=np.array([0.0, 0.0, 0.0]),
    dt=0.001,
    motor_time_constant=0.1,
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
    has_velocity_sensors=True,
    filter_cutoff_hz=50.0,
)

NOISE = NoiseForce(mean=0.00, std=0.03)
TARGET = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])

# ═══════════════════════════════════════════════════════════════════════════
# Точка входа
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ПИД-регулятор (работает сразу, без обучения)
    controller = PIDController(
        CONTROLLER_CONFIG,
        gains=np.array([80.0, 0.0, 30.0, -10.0, -15.0]),
    )

    print("Запуск GUI-симуляции...")
    print("  Управление: ПИД-регулятор")
    print("  Пробел — сброс, Q / ESC — выход")

    viewer = PendulumViewer(
        plant=ObjectOfControl(PLANT_CONFIG),
        sensor_config=SENSOR_CONFIG,
        noise=NOISE,
        target_state=TARGET,
        controller=controller,
    )
    viewer.use()

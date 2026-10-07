"""Запускает GUI-симуляцию перевёрнутого маятника."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from configs import PLANT_CONFIG, target
from packages.controllers.ppo import PPOController
from packages.simulation.co import (
    ControllerConfig,
    SensorConfig,
)
from packages.simulation.env.env import PendulumEnv
from packages.simulation.gui import PendulumViewer
from PPO_train import reward_f, terminate_condition, truncated_condition

# Validation checkpoint used by the GUI. This explicitly overrides the
# training output name imported from PPO_train.py.
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME = str(PROJECT_ROOT / "checkpoints/ppo/validation/model_1900000")

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


def save_reward_comparison(
    output_path: str | Path = PROJECT_ROOT / "plots" / "reward_comparison.png",
    angular_velocities: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0),
    samples: int = 500,
) -> Path:
    """Сохраняет графики полной и потенциальной энергии без открытия окна."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if not angular_velocities:
        raise ValueError("angular_velocities must not be empty")

    angles = np.linspace(0.0, 2.0 * np.pi, samples)
    target_state = target(0.0)

    mass = float(PLANT_CONFIG.m1)
    half_length = float(PLANT_CONFIG.L1)
    gravity = abs(float(PLANT_CONFIG.g))
    inertia_about_pivot = float(PLANT_CONFIG.J1) + mass * half_length**2
    potential_energy = mass * gravity * half_length * (1.0 - np.cos(angles))

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(angular_velocities)))

    for velocity, color in zip(angular_velocities, colors, strict=True):
        states = np.zeros((samples, 8), dtype=np.float64)
        states[:, 1] = np.cos(angles)
        states[:, 6] = velocity
        rewards = np.array([reward_f(state, target_state) for state in states])
        axes[0].plot(
            np.degrees(angles),
            rewards,
            color=color,
            linewidth=2.2,
            label=rf"$\dot{{\theta}}_1 = {velocity:g}$ рад/с",
        )

    axes[0].set_title("Текущая награда: полная энергия", pad=12)
    axes[0].set_ylabel("Энергия, Дж")
    axes[0].legend(frameon=False)

    axes[1].plot(
        np.degrees(angles),
        potential_energy,
        color="#d1495b",
        linewidth=2.8,
        label="Потенциальная энергия",
    )
    axes[1].set_title("Награда только за потенциальную энергию", pad=12)
    axes[1].set_ylabel("Энергия, Дж")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.set_xlabel(r"Угол маятника $\theta_1$, градусы")
        axis.set_xlim(0.0, 360.0)
        axis.set_xticks(np.arange(0.0, 361.0, 45.0))
        axis.grid(True, linestyle=":", alpha=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axvline(180.0, color="#555555", linestyle="--", linewidth=1, alpha=0.6)

    figure.suptitle("Сравнение функций награды для однозвенного маятника", fontsize=15)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    """Создаёт среду и контроллер, затем запускает окно симуляции."""
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
    if Path(f"{MODEL_NAME}.zip").exists():
        controller = PPOController.load(MODEL_NAME)
        print(f"Загружена модель: {MODEL_NAME}")
    else:
        print(f"Модель {MODEL_NAME} не найдена — ручное управление (←/→).")

    viewer = PendulumViewer(env=env, controller=controller)
    viewer.use()


if __name__ == "__main__":
    main()

"""Headless plot creator for the inverted pendulum.

Симулирует маятник ~10 секунд и сохраняет график cos(theta1).
Окна/GUI не открываются.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configs import PLANT_CONFIG, target
from packages.controllers.ppo import PPOController
from packages.simulation.co import ControllerConfig, SensorConfig
from packages.simulation.env.env import PendulumEnv
from PPO_train import reward_f, terminate_condition, truncated_condition


PROJECT_ROOT = Path(__file__).resolve().parent


def _default_sensor_config() -> SensorConfig:
    return SensorConfig(
        encoder_resolution_1=4096,
        encoder_resolution_2=4096,
        cart_sensor_resolution=0.0001,
        noise_std_q=(0.0005, 0.002, 0.002),
        noise_std_dq=(0.005, 0.01, 0.01),
    )


def _default_controller_config() -> ControllerConfig:
    return ControllerConfig(
        dt=0.01,
        max_force=24.0,
        has_velocity_sensors=False,
        filter_cutoff_hz=50.0,
    )


def simulate_and_save_cos_plot(
    *,
    model_name: str | Path = PROJECT_ROOT
    / "checkpoints/ppo/validation/model_1900000",
    max_real_time_s: float = 10.0,
    output_path: str | Path = PROJECT_ROOT / "plots" / "cos_theta1.png",
    render_every: int = 1,
) -> Path:
    """Симулирует до max_real_time_s и сохраняет график cos(theta1).

    По смыслу "реального времени" это означает: останавливаемся по сумме
    шагов среды (env.dt внутри контроллера) пока достигнут лимит.
    В GUI аналогичная величина `_sim_time`.
    """

    sensor_config = _default_sensor_config()
    controller_config = _default_controller_config()

    # env ожидает целевую функцию: target(t) -> ndarray
    env = PendulumEnv(
        PLANT_CONFIG,
        sensor_config,
        reward_f,
        terminate_condition,
        controller_config,
        target,
        truncated_condition,
    )

    controller_path_base = str(model_name)
    controller = None
    if Path(f"{controller_path_base}.zip").exists() or Path(
        f"{controller_path_base}_model.zip"
    ).exists():
        controller = PPOController.load(controller_path_base)
        print(f"Загружена модель: {controller_path_base}")
    else:
        raise FileNotFoundError(
            f"Модель не найдена: {controller_path_base}.zip или *_model.zip"
        )

    obs, _ = env.reset()

    # В телеметрии индексы зависят от env/plant. В GUI-сценариях cos(theta1)
    # используется как self._plant.get_telemetry()[:3] и графики рисуются через deque.
    # В run_gui.py/cfg обычно cos(theta1) соответствует obs[1].
    # Это соответствует тому, как в save_reward_comparison собирают cos(theta).
    COS_INDEX = 1

    cos_hist: deque[float] = deque()
    time_hist: deque[float] = deque()

    sim_dt = float(controller_config.dt)
    max_steps = int(math.ceil(max_real_time_s / sim_dt))
    t = 0.0

    for step in range(max_steps):
        # env obs shape: [telemetry(14?) + target(6)]
        # PPOController.action ожидает state_target: np.ndarray той же формы,
        # которую policy использует. В GUI он передаёт state_target целиком.
        action = controller.action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        _ = reward

        if step % render_every == 0:
            cos_val = float(np.asarray(obs, dtype=float)[COS_INDEX])
            cos_hist.append(cos_val)
            time_hist.append(t)

        t += sim_dt
        if terminated or truncated:
            break

    time_arr = np.asarray(time_hist, dtype=float)
    cos_arr = np.asarray(cos_hist, dtype=float)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11.5, 4.8), constrained_layout=True)
    plt.plot(time_arr, cos_arr, linewidth=2.5, color="#1f77b4")
    plt.title("cos(theta1) vs time")
    plt.xlabel("time, s")
    plt.ylabel("cos(theta1)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.ylim(-1.05, 1.05)
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved plot: {output_path}")
    return output_path


def main() -> None:
    simulate_and_save_cos_plot()


if __name__ == "__main__":
    main()

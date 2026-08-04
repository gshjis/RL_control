"""Профилирование обучения PPO-агента.

Запуск:
    poetry run python profiling/profile_ppo_train.py

Результаты:
    - profiling_outputs/*.pstats (cProfile)
"""

from __future__ import annotations

import sys
import time
import cProfile
import pstats
from pathlib import Path

import numpy as np

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from packages.controllers.PPO import PPOConfig, PPOController
from packages.simulation.CO.datatypes import (
    ControllerConfig,
    NoiseForce,
    PlantConfig,
    SensorConfig,
)


def main() -> None:
    out_dir = Path("profiling_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Конфигурации ────────────────────────────────────────────────────
    plant_cfg = PlantConfig(
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
        dt=0.0001,
        motor_time_constant=0.05,
    )
    sensor_cfg = SensorConfig(
        encoder_resolution_1=4096,
        encoder_resolution_2=4096,
        cart_sensor_resolution=0.0001,
        noise_std_q=(0.0005, 0.002, 0.002),
        noise_std_dq=(0.005, 0.01, 0.01),
    )
    controller_cfg = ControllerConfig(
        dt=0.01,
        max_force=24.0,
        has_velocity_sensors=False,
        filter_cutoff_hz=50.0,
    )
    noise = NoiseForce(mean=0.0, std=0.03)
    target = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])

    # ── Создание контроллера ────────────────────────────────────────────
    ppo_cfg = PPOConfig(
        total_timesteps=4096,
        n_steps=1024,
        net_arch=[128, 128],
    )
    controller = PPOController(
        ppo_config=ppo_cfg,
        controller_config=controller_cfg,
    )

    # ── Профилирование ──────────────────────────────────────────────────
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()

    try:
        controller.train(
            plant_config=plant_cfg,
            sensor_config=sensor_cfg,
            noise=noise,
            target_state=target,
        )
    finally:
        profiler.disable()

    elapsed = time.perf_counter() - t0

    print(f"\nPPO training ({ppo_cfg.total_timesteps} timesteps) finished in {elapsed:.3f} sec")
    print(f"Скорость: {ppo_cfg.total_timesteps / elapsed:.0f} timesteps/сек")

    # ── Сохранение результатов ──────────────────────────────────────────
    pstats_path = out_dir / f"ppo_train_{int(t0)}.pstats"
    profiler.dump_stats(str(pstats_path))

    ps = pstats.Stats(str(pstats_path))
    ps.strip_dirs().sort_stats("cumtime").print_stats(30)

    # Дополнительно: топ по внутреннему времени
    print("\n─── Топ по внутреннему времени (без учёта подвызовов) ───")
    ps.strip_dirs().sort_stats("time").print_stats(15)


if __name__ == "__main__":
    main()

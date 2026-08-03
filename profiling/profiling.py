"""Профилирование PendulumEnv (gym-обёртка симуляции маятника).

Запуск:
    poetry run python profiling/profiling.py

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

from packages.simulation.CO.datatypes import NoiseForce, PlantConfig, SensorConfig
from packages.simulation.ENV.env import PendulumEnv


def main() -> None:
    out_dir = Path("profiling_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Конфигурации ────────────────────────────────────────────────────
    plant_cfg = PlantConfig(
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
    sensor_cfg = SensorConfig(
        encoder_resolution_1=4096,
        encoder_resolution_2=4096,
        cart_sensor_resolution=0.0001,
        noise_std_q=(0.0005, 0.002, 0.002),
        noise_std_dq=(0.005, 0.01, 0.01),
    )
    noise = NoiseForce(mean=0.0, std=0.03)
    target = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])

    # ── Создание среды ──────────────────────────────────────────────────
    env = PendulumEnv(
        plant_config=plant_cfg,
        sensor_config=sensor_cfg,
        noise_force=noise,
        target_state=target,
        max_force=24.0,
        max_episode_steps=2000,
        dt_control=0.005,
    )

    # ── Профилирование ──────────────────────────────────────────────────
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()

    try:
        n_episodes = 5
        n_steps_per_episode = 500

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            for step in range(n_steps_per_episode):
                # Случайное действие (как в начале обучения)
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

    finally:
        profiler.disable()

    elapsed = time.perf_counter() - t0
    total_steps = n_episodes * n_steps_per_episode
    print(f"\nPendulumEnv: {n_episodes} эпизодов × до {n_steps_per_episode} шагов")
    print(f"Всего шагов: ~{total_steps}")
    print(f"Время: {elapsed:.3f} сек")
    print(f"Скорость: {total_steps / elapsed:.0f} шагов/сек")

    # ── Сохранение результатов ──────────────────────────────────────────
    pstats_path = out_dir / f"pendulum_env_{int(t0)}.pstats"
    profiler.dump_stats(str(pstats_path))

    ps = pstats.Stats(str(pstats_path))
    ps.strip_dirs().sort_stats("cumtime").print_stats(30)

    # Дополнительно: топ по времени в одном методе
    print("\n─── Топ по внутреннему времени (без учёта подвызовов) ───")
    ps.strip_dirs().sort_stats("time").print_stats(15)

    env.close()


if __name__ == "__main__":
    main()

"""Пресеты экспериментов — единый источник правды для всех точек входа.

Раньше одна и та же тройка конфигураций (растение / датчики / контроллер)
была продублирована в корневом конфиг-модуле, ``run_gui.py``,
``plot_creator.py`` и ``profiling`` с риском разъехаться. Теперь каждая
точка входа берёт конфигурацию отсюда.
"""

from __future__ import annotations

import numpy as np

from packages.controllers.ppo_config import PPOConfig
from packages.experiment import ExperimentConfig
from packages.simulation.co.datatypes import (
    ControllerConfig,
    PlantConfig,
    SensorConfig,
)
from packages.simulation.co.features import get_obs_dim


def _default_ppo_config(plant: PlantConfig, controller: ControllerConfig) -> PPOConfig:
    """Возвращает PPOConfig с net_arch, адаптированным под размерность наблюдения."""
    obs_dim = get_obs_dim(plant, controller)
    # Эвристика: размер скрытого слоя ~ 4-8x от входа, но не более 128
    hidden = min(max(obs_dim * 6, 32), 128)
    # Для малых входов (4-5) — 1 слой, для больших (7-8) — 2 слоя
    net_arch = [hidden] if obs_dim <= 5 else [hidden, hidden]

    return PPOConfig(
        total_timesteps=3_500_000,
        eval_freq=100_000,
        n_eval_episodes=2,
        max_episode_steps=2_000,
        net_arch=net_arch,
    )


def _base_plant(**overrides: object) -> PlantConfig:
    """Базовая конфигурация растения с переопределениями."""
    defaults: dict[str, object] = {
        "M": 1.0,
        "m1": 0.05,
        "l1": 0.2,
        "m2": 0.0,
        "l2": 0.0,
        "g": -9.81,
        "b_c": 0.01,
        "b_1": 0.001,
        "b_2": 0.001,
        "single_pendulum_mode": True,
        "init_q_stats": np.array([[0.0, np.pi, 0.0], [0.01, 0.00, 0.01]]),
        "init_dq_stats": np.array([[0.0, 0.0, 0.0], [0.0, 0.00, 0.01]]),
        "dt": 0.001,
        "motor_time_constant": 0.01,
    }
    defaults.update(overrides)
    return PlantConfig(**defaults)  # type: ignore[arg-type]


def _base_controller(**overrides: object) -> ControllerConfig:
    """Базовая конфигурация контроллера с переопределениями."""
    defaults: dict[str, object] = {
        "dt": 0.01,
        "max_force": 24.0,
        "has_velocity_sensors": False,
        "filter_cutoff_hz": 50.0,
    }
    defaults.update(overrides)
    return ControllerConfig(**defaults)  # type: ignore[arg-type]


def _base_sensor() -> SensorConfig:
    """Базовая конфигурация датчика."""
    return SensorConfig(
        encoder_resolution_1=4096,
        encoder_resolution_2=4096,
        cart_sensor_resolution=0.0001,
        noise_std_q=(0.0005, 0.002, 0.002),
        noise_std_dq=(0.005, 0.01, 0.01),
        exact_mode=True,
    )


def _make_experiment(
    plant: PlantConfig,
    controller: ControllerConfig,
    name: str = "ppo_upright",
) -> ExperimentConfig:
    """Собирает ExperimentConfig с автоматическим PPOConfig."""
    return ExperimentConfig(
        name=name,
        plant=plant,
        sensor=_base_sensor(),
        controller=controller,
        ppo=_default_ppo_config(plant, controller),
        n_envs=10,
    )


# ── 4 основных кейса ────────────────────────────────────────────────────────

def single_pendulum_no_pos_control(**plant_overrides: object) -> ExperimentConfig:
    """Одно звено, без позиционного управления: 4 признака [sinθ1, cosθ1, dθ1, dx]."""
    plant = _base_plant(single_pendulum_mode=True, **plant_overrides)
    controller = _base_controller(position_controlled=False)
    return _make_experiment(plant, controller, name="ppo_single_no_pos")


def single_pendulum_pos_control(**plant_overrides: object) -> ExperimentConfig:
    """Одно звено, с позиционным управлением: 5 признаков + x_err."""
    plant = _base_plant(single_pendulum_mode=True, **plant_overrides)
    controller = _base_controller(position_controlled=True)
    return _make_experiment(plant, controller, name="ppo_single_pos")


def double_pendulum_no_pos_control(**plant_overrides: object) -> ExperimentConfig:
    """Два звена, без позиционного управления: 7 признаков + sinθ2, cosθ2, dθ2."""
    plant = _base_plant(
        single_pendulum_mode=False, m2=0.05, l2=0.2, **plant_overrides
    )
    controller = _base_controller(position_controlled=False)
    return _make_experiment(plant, controller, name="ppo_double_no_pos")


def double_pendulum_pos_control(**plant_overrides: object) -> ExperimentConfig:
    """Два звена, с позиционным управлением: 8 признаков + x_err."""
    plant = _base_plant(
        single_pendulum_mode=False, m2=0.05, l2=0.2, **plant_overrides
    )
    controller = _base_controller(position_controlled=True)
    return _make_experiment(plant, controller, name="ppo_double_pos")


def default() -> ExperimentConfig:
    """Возвращает базовую конфигурацию обучения PPO на одно-звенном маятнике (как single_pendulum_no_pos_control)."""
    return single_pendulum_no_pos_control()


__all__ = [
    "default",
    "single_pendulum_no_pos_control",
    "single_pendulum_pos_control",
    "double_pendulum_no_pos_control",
    "double_pendulum_pos_control",
]

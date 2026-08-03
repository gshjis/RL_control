from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOConfig:
    """Конфигурация гиперпараметров PPO-алгоритма."""

    # Параметры сети
    policy: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [64, 64])

    # Параметры обучения
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5

    # Параметры среды
    total_timesteps: int = 500_000
    max_episode_steps: int = 10000
    dt_control: float = 0.001
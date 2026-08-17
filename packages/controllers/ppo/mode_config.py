from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    """Конфигурация гиперпараметров PPO-алгоритма."""

    policy: str = "MlpPolicy"
    net_arch: list[int] = field(default_factory=lambda: [64, 64])

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 8

    # Swing-up takes several controller ticks, so use a longer planning
    # horizon than the default 0.99 discount.
    gamma: float = 0.999
    gae_lambda: float = 0.97

    clip_range: float = 0.2
    ent_coef: float = 0.05

    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    total_timesteps: int = 300_000
    max_episode_steps: int = 10000
    seed: int = 42

    eval_freq: int = 10_000
    n_eval_episodes: int = 10

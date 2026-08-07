from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class PPOConfig:
    """Конфигурация гиперпараметров PPO-алгоритма."""

    policy: str = "MlpPolicy"
    net_arch: list[int] = field(default_factory=lambda: [64, 64, 64])

    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 8                  
    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    clip_range: float = 0.15          
    ent_coef: float = 0.05            
    
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    total_timesteps: int = 300_000     
    max_episode_steps: int = 1024
    seed: int = 42

    eval_freq: int = 10_000
    n_eval_episodes: int = 10

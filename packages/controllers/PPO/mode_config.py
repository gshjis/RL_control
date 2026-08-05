from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    """Конфигурация гиперпараметров PPO-алгоритма."""

    # Параметры сети
    policy: str = "MlpPolicy"
    net_arch: list[int] = field(default_factory=lambda: [64, 64])

    # Параметры обучения (оптимизированы для CartPole/кастомной среды)
    learning_rate: float = 3e-4          # Стандарт, работает хорошо
    n_steps: int = 512                   # 2048 → 512 (главное изменение!)
    batch_size: int = 64                 # 512 * n_envs / 64 = целое число
    n_epochs: int = 5                    # 10 → 5 (стабильнее)
    gamma: float = 0.99                  # Стандарт
    gae_lambda: float = 0.95             # Стандарт
    clip_range: float = 0.2              # Стандарт
    ent_coef: float = 0.01               # 0.1 → 0.01 (меньше случайности)
    vf_coef: float = 0.5                 # Явно добавляем (было неявно)
    max_grad_norm: float = 0.5           # Стандарт

    # Параметры среды
    total_timesteps: int = 500_000       # Достаточно для обучения
    max_episode_steps: int = 10000       # Максимум в вашей среде
    seed: int = 42

    # Параметры валидации (увеличены для стабильности)
    eval_freq: int = 10_000              # Проверка каждые 10K шагов
    n_eval_episodes: int = 10            # 5 → 10 (стабильнее оценка)
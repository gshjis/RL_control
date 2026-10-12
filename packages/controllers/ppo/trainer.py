"""Обучение PPO: сборка пулов сред, запуск обучения и валидация."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
    sync_envs_normalization,
)

from packages.controllers.ppo.ppo import PPOController
from packages.controllers.ppo_config import PPOConfig
from packages.simulation.co.datatypes import ControllerConfig

# Фабрика сред: каждый вызов возвращает новую (не используемую) среду.
EnvFactory = Callable[[], gym.Env]


class ValidationCallback(BaseCallback):
    """Оценивает PPO в отдельной среде и сохраняет результаты проверки."""

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        max_episode_steps: int = 1000,
        checkpoint_dir: str = "checkpoints/ppo/validation",
        verbose: int = 1,
    ) -> None:
        """Принимает среду и параметры оценки, сохраняет их для callback."""
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.max_episode_steps = int(max_episode_steps)
        self.checkpoint_dir = checkpoint_dir
        self._last_eval_timestep = 0

    def _on_step(self) -> bool:
        """Проверяет частоту оценки и возвращает признак продолжения обучения."""
        # n_calls counts callback invocations, while num_timesteps counts
        # actual transitions and includes all parallel environments. Schedule
        # validation by the latter so eval_freq means training timesteps.
        if self.num_timesteps - self._last_eval_timestep >= self.eval_freq:
            self._last_eval_timestep = self.num_timesteps
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        """Оценивает политику, записывает награду и сохраняет checkpoint."""
        train_vec_norm = self.model.get_vec_normalize_env()
        if train_vec_norm is not None:
            sync_envs_normalization(train_vec_norm, self.eval_env)

        episode_rewards: list[float] = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            while not done and steps < self.max_episode_steps:
                actions, _ = self.model.predict(
                    cast(np.ndarray, obs), deterministic=True
                )
                obs, rewards, dones, _ = self.eval_env.step(actions)
                ep_reward += float(rewards[0])
                steps += 1
                done = bool(dones[0])
            episode_rewards.append(ep_reward)

        mean_reward = float(np.mean(episode_rewards))
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.dump(self.num_timesteps)

        # Сохраняем отдельный snapshot после каждой валидации. Помимо весов
        # политики сохраняем статистику VecNormalize, иначе модель нельзя
        # корректно использовать с теми же нормализованными наблюдениями.
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"model_{self.num_timesteps}"
        )
        self.model.save(checkpoint_path)
        train_vec_norm = self.model.get_vec_normalize_env()
        if train_vec_norm is not None:
            train_vec_norm.save(f"{checkpoint_path}_vecnormalize.pkl")

        if self.verbose:
            print(
                f"[Eval] timesteps={self.num_timesteps} | "
                f"mean_reward={mean_reward:.4f} | saved={checkpoint_path}.zip"
            )


class PPOTrainer:
    """Собирает пулы сред и обучает политику PPO.

    Тренер — единственное место, где встречаются SB3, пулы сред и
    валидация. Контроллер, который он возвращает, умеет только выполнять
    политику (``act``) и сохранять/загружать веса.
    """

    def __init__(
        self,
        *,
        ppo_config: PPOConfig,
        controller_config: ControllerConfig,
        env_factory: EnvFactory,
        n_envs: int = 1,
        checkpoint_dir: str = "checkpoints/ppo/validation",
    ) -> None:
        """Принимает конфигурации и фабрику сред для обучения."""
        if n_envs < 1:
            raise ValueError("n_envs must be positive")
        self._ppo_config = ppo_config
        self._controller_config = controller_config
        self._env_factory = env_factory
        self._n_envs = int(n_envs)
        self._checkpoint_dir = checkpoint_dir

    def fit(self, verbose: int = 1) -> PPOController:
        """Обучает политику и возвращает контроллер, готовый к применению."""
        train_vec_env = VecNormalize(
            SubprocVecEnv([self._env_factory] * self._n_envs),
            norm_obs=True,
            norm_reward=False,
        )
        model = SB3_PPO(
            self._ppo_config.policy,
            train_vec_env,
            learning_rate=self._ppo_config.learning_rate,
            n_steps=self._ppo_config.n_steps,
            batch_size=self._ppo_config.batch_size,
            n_epochs=self._ppo_config.n_epochs,
            gamma=self._ppo_config.gamma,
            gae_lambda=self._ppo_config.gae_lambda,
            clip_range=self._ppo_config.clip_range,
            ent_coef=self._ppo_config.ent_coef,
            vf_coef=self._ppo_config.vf_coef,
            max_grad_norm=self._ppo_config.max_grad_norm,
            policy_kwargs={"net_arch": self._ppo_config.net_arch},
            verbose=verbose,
            seed=self._ppo_config.seed,
        )

        # Валидационная среда (отдельная, обёрнутая в VecNormalize).
        eval_vec_env = VecNormalize(
            DummyVecEnv([self._env_factory]),
            norm_obs=True,
            # Keep validation rewards raw so mean_reward reflects the actual
            # reward function instead of VecNormalize's running statistics.
            norm_reward=False,
        )
        eval_callback = ValidationCallback(
            eval_env=eval_vec_env,
            eval_freq=self._ppo_config.eval_freq,
            n_eval_episodes=self._ppo_config.n_eval_episodes,
            max_episode_steps=self._ppo_config.max_episode_steps,
            checkpoint_dir=self._checkpoint_dir,
            verbose=verbose,
        )

        model.learn(
            total_timesteps=self._ppo_config.total_timesteps,
            callback=eval_callback,
        )

        return PPOController.from_model(
            model,
            vec_normalize=train_vec_env,
            ppo_config=self._ppo_config,
            controller_config=self._controller_config,
        )

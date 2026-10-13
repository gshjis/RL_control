"""Обучение PPO: сборка пулов сред, запуск обучения и валидация."""

from __future__ import annotations

import os
from functools import partial
from typing import cast

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
from packages.experiment import EnvFactory
from packages.simulation.co.datatypes import ControllerConfig


class ValidationCallback(BaseCallback):
    """Оценивает PPO в отдельной среде и сохраняет результаты проверки.

    Валидация запускается только на эпизодах, стартующих в НИЗУ (theta=0, cos=1).
    Эпизоды, стартующие ВВЕРХУ (theta=pi, cos=-1), пропускаются — там тривиально.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        max_episode_steps: int = 1000,
        checkpoint_dir: str = "checkpoints/ppo/validation",
        verbose: int = 1,
        upright_cos_threshold: float = -0.5,  # cos(theta) < -0.5 => upright
    ) -> None:
        """Принимает среду и параметры оценки, сохраняет их для callback."""
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.max_episode_steps = int(max_episode_steps)
        self.checkpoint_dir = checkpoint_dir
        self._last_eval_timestep = 0
        self._upright_cos_threshold = upright_cos_threshold

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
            obs_any = self.eval_env.reset()
            # VecNormalize.reset() returns np.ndarray for vectorized envs
            obs_arr = obs_any[0] if isinstance(obs_any, tuple) else obs_any
            # Ensure it's an ndarray (VecNormalize with VecEnv returns np.ndarray)
            assert isinstance(obs_arr, np.ndarray), (
                f"Expected ndarray, got {type(obs_arr)}"
            )
            # Денормализуем наблюдение, чтобы получить RAW признаки
            # (VecNormalize хранит running mean/std и может вернуть исходные значения)
            vec_norm = cast(VecNormalize, self.eval_env)
            obs_raw = vec_norm.unnormalize_obs(obs_arr)
            # unnormalize_obs может вернуть dict для Dict observation spaces,
            # но у нас Box space -> всегда np.ndarray
            assert isinstance(obs_raw, np.ndarray), (
                f"Expected ndarray, got {type(obs_raw)}"
            )
            # Проверяем стартовое положение: cos(theta1) в RAW наблюдении
            # В features (4/5/7/8) cos_theta1 на индексе 1
            cos_theta1 = (
                float(obs_raw[0, 1]) if obs_raw.ndim == 2 else float(obs_raw[1])
            )

            # Пропускаем эпизоды, стартующие ВВЕРХУ (cos ≈ -1)
            if cos_theta1 < self._upright_cos_threshold:
                if self.verbose:
                    print(
                        f"  [Validation] Skip upright start: "
                        f"cos(theta1)={cos_theta1:.3f}"
                    )
                continue

            done = False
            ep_reward = 0.0
            steps = 0
            while not done and steps < self.max_episode_steps:
                actions, _ = self.model.predict(
                    cast(np.ndarray, obs_arr), deterministic=True
                )
                step_result = self.eval_env.step(actions)
                obs_next = step_result[0]
                rewards = step_result[1]
                dones = step_result[2]
                obs_arr = obs_next[0] if isinstance(obs_next, tuple) else obs_next
                assert isinstance(obs_arr, np.ndarray), (
                    f"Expected ndarray, got {type(obs_arr)}"
                )
                ep_reward += float(rewards[0])
                steps += 1
                done = bool(dones[0])
            episode_rewards.append(ep_reward)

        if not episode_rewards:
            if self.verbose:
                print("  [Validation] All episodes skipped (upright starts)")
            return

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
        # Ранг в пуле сдвигает seed каждой среды (см. make_env_factory):
        # параллельные эпизоды независимы, но весь запуск воспроизводим.
        train_vec_env = VecNormalize(
            SubprocVecEnv(
                [partial(self._env_factory, rank) for rank in range(self._n_envs)]
            ),
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
        # Ранг 0: eval всегда один и тот же детерминированный поток.
        eval_vec_env = VecNormalize(
            DummyVecEnv([partial(self._env_factory, 0)]),
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
            upright_cos_threshold=-0.5,
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

from __future__ import annotations

import dataclasses
import json
import os
from typing import cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    sync_envs_normalization,
)

from packages.controllers.PPO.mode_config import PPOConfig
from packages.simulation.CO import (
    Controller,
    ControllerConfig,
)
from packages.simulation.ENV.env_orcestrator import EnvOrchestrator


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


class _DummyEnv(gym.Env):
    """Минимальная среда для загрузки VecNormalize (нужен только venv)."""

    def __init__(self, observation_space, action_space) -> None:
        """Принимает пространства наблюдений и действий для загрузки статистики."""
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        """Принимает seed и options, возвращает случайное наблюдение."""
        return self.observation_space.sample(), {}

    def step(self, action):
        """Принимает действие, возвращает фиктивный результат шага."""
        return self.observation_space.sample(), 0.0, False, False, {}


class PPOController(Controller):
    """Управляет обучением, сохранением и применением политики PPO."""

    def __init__(
        self,
        ppo_config: PPOConfig,
        controller_config: ControllerConfig,
        model: SB3_PPO | None = None,
    ) -> None:
        """Принимает конфигурации PPO и контроллера, создаёт обёртку модели."""

        super().__init__(controller_config)
        self.name = "PPO"

        self._ppo_config = ppo_config
        self._controller_config = controller_config
        self._model: SB3_PPO | None = model

        self._vec_normalize: VecNormalize | None = None

    def save(self, name: str) -> None:
        """Принимает имя и сохраняет модель с конфигурацией."""
        if self._model is None:
            raise RuntimeError("Нет модели для сохранения. Сначала вызовите train().")

        self._model.save(f"{name}_model")
        if self._vec_normalize is not None:
            self._vec_normalize.save(f"{name}_vecnormalize.pkl")

        payload = {
            "ppo_config": dataclasses.asdict(self._ppo_config),
            "controller_config": dataclasses.asdict(self._controller_config),
        }
        with open(f"{name}_config.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, name: str) -> PPOController:
        """Принимает имя и возвращает восстановленный PPO-контроллер."""
        config_path = f"{name}_config.json"
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                payload = json.load(f)

            ppo_config = PPOConfig(**payload["ppo_config"])
            controller_config = ControllerConfig(**payload["controller_config"])
        else:
            # Validation snapshots contain the model and VecNormalize state,
            # but historically did not contain a config JSON.  The model
            # already stores its observation/action spaces, so defaults are
            # sufficient for inference and keep old snapshots loadable.
            print(
                f"Предупреждение: {config_path} не найден; "
                "загружаются конфигурации PPO по умолчанию."
            )
            ppo_config = PPOConfig()
            controller_config = ControllerConfig()

        # Final training checkpoints use ``<name>_model.zip`` while validation
        # snapshots use ``<name>.zip``. Support both naming conventions.
        model_path = name if os.path.exists(f"{name}.zip") else f"{name}_model"
        model = SB3_PPO.load(model_path)
        obj = cls(
            ppo_config=ppo_config,
            controller_config=controller_config,
            model=model,
        )

        vec_norm_path = f"{name}_vecnormalize.pkl"
        if os.path.exists(vec_norm_path):
            dummy_venv = DummyVecEnv(
                [lambda: _DummyEnv(model.observation_space, model.action_space)]
            )
            obj._vec_normalize = VecNormalize.load(vec_norm_path, dummy_venv)

        return obj

    def action(self, state_target: np.ndarray) -> np.ndarray:
        """Принимает состояние с целью, возвращает действие политики PPO."""
        if self._model is None:
            raise RuntimeError("PPO-модель не загружена. Вызовите train() или load().")

        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(state_target)
        else:
            obs = state_target
        actions, _ = self._model.predict(obs, deterministic=True)
        return actions

    def train(
        self,
        env_orchestrator: EnvOrchestrator,
    ) -> None:
        """Принимает оркестратор, обучает и периодически валидирует PPO-модель."""
        vec_env = VecNormalize(
            env_orchestrator._env_hub, norm_obs=True, norm_reward=False
        )
        self._vec_normalize = vec_env
        if self._model is None:
            self._model = SB3_PPO(
                self._ppo_config.policy,
                self._vec_normalize,
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
                verbose=1,
                seed=self._ppo_config.seed,
            )

        # Валидационная среда (отдельная, обёрнутая в VecNormalize).
        eval_env = VecNormalize(
            DummyVecEnv([env_orchestrator._make_env]),
            norm_obs=True,
            # Keep validation rewards raw so mean_reward reflects the actual
            # reward function instead of VecNormalize's running statistics.
            norm_reward=False,
        )
        eval_callback = ValidationCallback(
            eval_env=eval_env,
            eval_freq=self._ppo_config.eval_freq,
            n_eval_episodes=self._ppo_config.n_eval_episodes,
            max_episode_steps=self._ppo_config.max_episode_steps,
        )

        self._model.learn(
            total_timesteps=self._ppo_config.total_timesteps,
            callback=eval_callback,
        )

from __future__ import annotations

import dataclasses
import json
import os

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
    """
    Периодически оценивает текущую политику на отдельной валидационной среде
    и выводит среднюю награду.

    Parameters
    ----------
    eval_env : VecEnv
        Валидационная среда (обёрнутая в VecNormalize).
    eval_freq : int
        Как часто (в шагах) выполнять оценку.
    n_eval_episodes : int
        Сколько эпизодов прогонять за одну оценку.
    max_episode_steps : int
        Максимальная длина эпизода при оценке.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        max_episode_steps: int = 1000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.max_episode_steps = int(max_episode_steps)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self) -> None:
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
                actions, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = self.eval_env.step(actions)
                ep_reward += float(rewards[0])
                steps += 1
                done = bool(dones[0])
            episode_rewards.append(ep_reward)

        mean_reward = float(np.mean(episode_rewards))
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.dump(self.num_timesteps)
        if self.verbose:
            print(
                f"[Eval] timesteps={self.num_timesteps} | "
                f"mean_reward={mean_reward:.4f}"
            )


class _DummyEnv(gym.Env):
    """Минимальная среда для загрузки VecNormalize (нужен только venv)."""

    def __init__(self, observation_space, action_space) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


class PPOController(Controller):
    def __init__(
        self,
        ppo_config: PPOConfig,
        controller_config: ControllerConfig,
        model: SB3_PPO | None = None,
    ) -> None:

        super().__init__(controller_config)
        self.name = "PPO"

        self._ppo_config = ppo_config
        self._controller_config = controller_config
        self._model: SB3_PPO | None = model

        self._vec_normalize: VecNormalize | None = None

    def save(self, name: str) -> None:
        """
        Сохранить модель и нормализатор (VecNormalize) по имени.

        Создаются файлы:
        - ``<name>_model.zip`` — модель PPO;
        - ``<name>_vecnormalize.pkl`` — статистика нормализации;
        - ``<name>_config.json`` — конфиги (для восстановления при load).
        """
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
        """
        Загрузить модель и нормализатор (VecNormalize) по имени.

        Восстанавливает конфиги из ``<name>_config.json``, модель из
        ``<name>_model.zip`` и нормализатор из ``<name>_vecnormalize.pkl``
        (если он был сохранён).
        """
        with open(f"{name}_config.json", "r", encoding="utf-8") as f:
            payload = json.load(f)

        ppo_config = PPOConfig(**payload["ppo_config"])
        controller_config = ControllerConfig(**payload["controller_config"])

        model = SB3_PPO.load(f"{name}_model")
        obj = cls(
            ppo_config=ppo_config,
            controller_config=controller_config,
            model=model,
        )

        vec_norm_path = f"{name}_vecnormalize.pkl"
        if os.path.exists(vec_norm_path):
            dummy_venv = DummyVecEnv(
                [
                    lambda: _DummyEnv(
                        model.observation_space, model.action_space
                    )
                ]
            )
            obj._vec_normalize = VecNormalize.load(vec_norm_path, dummy_venv)

        return obj

    def action(self,state_target: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PPO-модель не загружена. Вызовите train() или load().")
        
        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(state_target)
        else:
            obs = state_target
        actions, _ = self._model.predict(obs, deterministic=True)
        return actions 

    def train(
        self, env_orchestrator: EnvOrchestrator,
        ) -> None:
        vec_env = VecNormalize(env_orchestrator._env_hub, norm_obs=True, norm_reward=True)
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
                policy_kwargs=dict(net_arch=self._ppo_config.net_arch),
                verbose=1,
                seed=self._ppo_config.seed,
            )

        # Валидационная среда (отдельная, обёрнутая в VecNormalize).
        eval_env = VecNormalize(
            DummyVecEnv([env_orchestrator._make_env]),
            norm_obs=True,
            norm_reward=True,
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

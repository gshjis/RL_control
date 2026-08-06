from __future__ import annotations

import dataclasses
import json
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)

from packages.controllers.PPO.mode_config import PPOConfig
from packages.simulation.CO import (
    Controller,
    ControllerConfig,
)
from packages.simulation.ENV.env_orcestrator import EnvOrchestrator


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
                "MlpPolicy",
                self._vec_normalize,
                learning_rate=self._ppo_config.learning_rate,
                n_steps=self._ppo_config.n_steps,
                batch_size=self._ppo_config.batch_size,
                n_epochs=self._ppo_config.n_epochs,
                gamma=self._ppo_config.gamma,
                verbose=1,
                seed=self._ppo_config.seed,
            )

        self._model.learn(
            total_timesteps=self._ppo_config.total_timesteps,
        )

from __future__ import annotations

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
        self._model: SB3_PPO | None = model

        self._vec_normalize: VecNormalize | None = None

    def action(self, s_clean: np.ndarray, target_state: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PPO-модель не загружена. Вызовите train() или load().")
        
        obs = np.concatenate([s_clean, target_state], axis=1)
        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(obs)
        actions, _ = self._model.predict(obs, deterministic=True)
        return actions 

    def train(
        self, env_orchestrator: EnvOrchestrator,
        ) -> None:
        vec_env = VecNormalize(env_orchestrator._env_hub, norm_obs=True, norm_reward=False)
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

        # Валидационная среда (отдельная, обёрнутая в VecNormalize).
        eval_env = VecNormalize(
            DummyVecEnv([env_orchestrator._make_env]),
            norm_obs=True,
            norm_reward=False,
        )
        # Обучить
        self._model.learn(
            total_timesteps=self._ppo_config.total_timesteps,
        )

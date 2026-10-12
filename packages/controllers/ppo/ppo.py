"""Инференс-обёртка политики PPO: действие, сохранение и загрузка."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from packages.controllers.base.base import Controller
from packages.controllers.ppo_config import PPOConfig
from packages.simulation.co.datatypes import ControllerConfig


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
    """Применяет обученную политику PPO к наблюдениям среды.

    Обучение вынесено в ``packages.controllers.ppo.trainer.PPOTrainer``:
    контроллер не знает ни о пулах сред, ни о callback'ах.
    """

    def __init__(
        self,
        ppo_config: PPOConfig,
        controller_config: ControllerConfig,
        model: SB3_PPO | None = None,
        vec_normalize: VecNormalize | None = None,
    ) -> None:
        """Принимает конфигурации PPO и контроллера, создаёт обёртку модели."""
        super().__init__(controller_config)
        self._name = "PPO"
        self._ppo_config = ppo_config
        self._model: SB3_PPO | None = model
        self._vec_normalize: VecNormalize | None = vec_normalize

    @classmethod
    def from_model(
        cls,
        model: SB3_PPO,
        *,
        vec_normalize: VecNormalize | None = None,
        ppo_config: PPOConfig | None = None,
        controller_config: ControllerConfig | None = None,
    ) -> PPOController:
        """Принимает обученную модель SB3, возвращает готовый контроллер."""
        return cls(
            ppo_config=ppo_config or PPOConfig(),
            controller_config=controller_config or ControllerConfig(),
            model=model,
            vec_normalize=vec_normalize,
        )

    @property
    def model(self) -> SB3_PPO | None:
        """Возвращает модель SB3 или None, если она ещё не обучена/загружена."""
        return self._model

    @property
    def vec_normalize(self) -> VecNormalize | None:
        """Возвращает нормализатор наблюдений, если он использовался при обучении."""
        return self._vec_normalize

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Принимает наблюдение среды, возвращает детерминированное действие."""
        if self._model is None:
            raise RuntimeError("PPO-модель не загружена. Обучите её через PPOTrainer.")

        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(observation)
        else:
            obs = observation
        actions, _ = self._model.predict(cast(np.ndarray, obs), deterministic=True)
        return actions

    def save(self, name: str) -> None:
        """Принимает имя и сохраняет модель, нормализатор и конфигурации."""
        if self._model is None:
            raise RuntimeError("Нет модели для сохранения. Сначала обучите PPOTrainer.")

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

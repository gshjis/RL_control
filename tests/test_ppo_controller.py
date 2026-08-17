from pathlib import Path
from typing import cast

import numpy as np
import pytest
from stable_baselines3.common.vec_env import VecEnv

from packages.controllers.ppo import PPOConfig, PPOController
from packages.controllers.ppo.ppo import ValidationCallback
from packages.simulation.co.datatypes import ControllerConfig


class DummyModel:
    """Минимальная модель для проверки вызова политики PPO."""

    def predict(
        self, observation: np.ndarray, deterministic: bool
    ) -> tuple[np.ndarray, None]:
        """Возвращает фиксированное действие."""
        assert observation.shape == (1,)
        assert deterministic is True
        return np.array([0.25], dtype=np.float64), None

    def save(self, path: str) -> None:
        """Создаёт фиктивный файл модели."""
        Path(f"{path}.zip").write_bytes(b"model")


def make_controller() -> PPOController:
    return PPOController(PPOConfig(), ControllerConfig())


def test_ppo_action_requires_model() -> None:
    controller = make_controller()

    with pytest.raises(RuntimeError, match="PPO-модель не загружена"):
        controller.action(np.zeros(1))


def test_ppo_action_uses_loaded_model() -> None:
    controller = make_controller()
    controller._model = DummyModel()  # type: ignore[assignment]

    action = controller.action(np.zeros(1))

    np.testing.assert_allclose(action, [0.25])


def test_ppo_save_requires_model(tmp_path) -> None:
    controller = make_controller()

    with pytest.raises(RuntimeError, match="Нет модели для сохранения"):
        controller.save(str(tmp_path / "model"))


def test_ppo_save_writes_model_and_config(tmp_path) -> None:
    controller = make_controller()
    controller._model = DummyModel()  # type: ignore[assignment]
    name = tmp_path / "model"

    controller.save(str(name))

    assert (tmp_path / "model_model.zip").exists()
    assert (tmp_path / "model_config.json").exists()


def test_validation_callback_runs_on_schedule(monkeypatch) -> None:
    callback = ValidationCallback(eval_env=cast(VecEnv, None), eval_freq=10)
    callback.num_timesteps = 10
    evaluated = False

    def evaluate() -> None:
        nonlocal evaluated
        evaluated = True

    monkeypatch.setattr(callback, "_evaluate", evaluate)

    assert callback._on_step()
    assert evaluated

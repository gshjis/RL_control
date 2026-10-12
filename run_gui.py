"""Запускает GUI-симуляцию перевёрнутого маятника."""

from __future__ import annotations

from pathlib import Path

from packages.controllers.ppo import PPOController
from packages.experiment import build_env
from packages.presets import default
from packages.simulation.gui import PendulumViewer

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = str(PROJECT_ROOT / "checkpoints/ppo/validation/model_1900000")


def load_controller(model_name: str) -> PPOController | None:
    """Загружает чекпоинт; если его нет, возвращает None (ручной режим)."""
    if Path(f"{model_name}.zip").exists():
        print(f"Загружена модель: {model_name}")
        return PPOController.load(model_name)

    print(f"Модель {model_name} не найдена — ручное управление (←/→).")
    return None


def main(model_name: str = DEFAULT_MODEL_NAME) -> None:
    """Создаёт среду и контроллер, затем запускает окно симуляции."""
    experiment = default()
    env = build_env(experiment)
    controller = load_controller(model_name)

    viewer = PendulumViewer(env=env, controller=controller)
    viewer.use()


if __name__ == "__main__":
    main()

"""Загрузка чекпоинтов PPO."""

from __future__ import annotations

from pathlib import Path

from packages.controllers.ppo.ppo import PPOController


def load_controller(model_name: str | Path) -> PPOController:
    """Принимает имя чекпоинта и возвращает обученный контроллер.

    Поддерживаются оба исторических соглашения об именовании:
    ``<name>.zip`` (снимки валидации) и ``<name>_model.zip`` (итоговые
    чекпоинты обучения). Если модели нет — бросает FileNotFoundError.
    """
    base = str(model_name)
    if Path(f"{base}.zip").exists() or Path(f"{base}_model.zip").exists():
        return PPOController.load(base)

    raise FileNotFoundError(f"Модель не найдена: {base}.zip или {base}_model.zip")

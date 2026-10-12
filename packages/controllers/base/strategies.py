"""Именованные стратегии награды и условий завершения эпизода.

Названные стратегии (а не произвольные callable) делают описание эксперимента
сериализуемым: набор имён можно сохранить рядом с чекпоинтом и восстановить
конфигурацию без повторного программирования функций награды.
"""

from __future__ import annotations

from typing import Literal

from packages.simulation.co.datatypes import PlantConfig

from .cost_functions import (
    RewardFn,
    TerminateFn,
    TruncateFn,
    make_energy_potential_reward,
    make_energy_reward,
)
from .terminate_conditions import terminate_condition
from .truncate_conditions import truncated_condition

RewardSpec = Literal["energy", "energy_potential"]
TerminateSpec = Literal["cart_limit"]
TruncateSpec = Literal["steps"]

# Заводы стратегий: имя -> фабрика. Добавление новой награды — одна строка
# здесь плюс Literal выше.
_REWARD_FACTORIES = {
    "energy": make_energy_reward,
    "energy_potential": make_energy_potential_reward,
}

_TERMINATE_FACTORIES = {
    "cart_limit": lambda: terminate_condition,
}

_TRUNCATE_FACTORIES = {
    "steps": lambda: truncated_condition,
}


def _unknown(name: str, registry: dict[str, object]) -> ValueError:
    """Возвращает понятную ошибку с перечнем доступных имён."""
    available = ", ".join(sorted(registry))
    return ValueError(f"Неизвестная стратегия {name!r}; доступны: {available}")


def build_reward(spec: str, plant_config: PlantConfig) -> RewardFn:
    """Принимает имя стратегии награды и параметры маятника, возвращает reward."""
    try:
        factory = _REWARD_FACTORIES[spec]
    except KeyError as exc:
        raise _unknown(spec, _REWARD_FACTORIES) from exc
    return factory(plant_config)


def build_terminate(spec: str) -> TerminateFn:
    """Принимает имя стратегии terminate и возвращает функцию условия."""
    try:
        factory = _TERMINATE_FACTORIES[spec]
    except KeyError as exc:
        raise _unknown(spec, _TERMINATE_FACTORIES) from exc
    return factory()


def build_truncate(spec: str) -> TruncateFn:
    """Принимает имя стратегии truncate и возвращает функцию условия."""
    try:
        factory = _TRUNCATE_FACTORIES[spec]
    except KeyError as exc:
        raise _unknown(spec, _TRUNCATE_FACTORIES) from exc
    return factory()


__all__ = [
    "RewardSpec",
    "TerminateSpec",
    "TruncateSpec",
    "build_reward",
    "build_terminate",
    "build_truncate",
]

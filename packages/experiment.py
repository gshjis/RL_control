"""Единый описатель эксперимента.

``ExperimentConfig`` — замороженный набор конфигураций плюс именованные
стратегии награды/условий. Такой объект сериализуется целиком, поэтому
каждый запуск сохраняет рядом с чекпоинтом точную копию того, чем обучался
агент (см. ``PPOController.save`` и CLI-обёртки).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from packages.controllers.base.cost_functions import (
    RewardFn,
    TerminateFn,
    TruncateFn,
)
from packages.controllers.base.strategies import (
    RewardSpec,
    TerminateSpec,
    TruncateSpec,
    build_reward,
    build_terminate,
    build_truncate,
)
from packages.controllers.ppo_config import PPOConfig
from packages.simulation.co.datatypes import (
    ControllerConfig,
    PlantConfig,
    SensorConfig,
)

if TYPE_CHECKING:
    # Только для аннотаций: тяжёлые импорты Gymnasium-слоя отложены.
    import gymnasium as gym

    from packages.simulation.env.env import PendulumEnv

EnvFactory = Callable[[], gym.Env]

TargetFn = Callable[[float], np.ndarray]
TargetSpec = Literal["upright"]


def upright_target(time: float) -> np.ndarray:
    """Принимает время и возвращает цель «первый маятник вертикально».

    Компоненты: [x, cos t1, sin t1, cos t2, sin t2, dx]. Вертикаль первого
    звена — t1 = pi (cos = -1), второе звено направлено вдоль первого
    (t2 = 0, cos = 1). Позиция тележки и скорость цель не ограничивают.
    """
    del time
    return np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])


_TARGET_FACTORIES: dict[str, TargetFn] = {
    "upright": upright_target,
}


def build_target(spec: str) -> TargetFn:
    """Принимает имя стратегии цели и возвращает функцию target(time)."""
    try:
        return _TARGET_FACTORIES[spec]
    except KeyError as exc:
        available = ", ".join(sorted(_TARGET_FACTORIES))
        raise ValueError(
            f"Неизвестная стратегия цели {spec!r}; доступны: {available}"
        ) from exc


@dataclass(frozen=True)
class ExperimentConfig:
    """Описывает эксперимент целиком: модель, датчики, контроллер и стратегии."""

    name: str = "default"
    plant: PlantConfig = field(default_factory=PlantConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    reward: RewardSpec = "energy"
    terminate: TerminateSpec = "cart_limit"
    truncate: TruncateSpec = "steps"
    target: TargetSpec = "upright"

    n_envs: int = 10

    # ── Фабрики: имя стратегии -> рабочая функция ──────────────────────────
    def make_reward(self) -> RewardFn:
        """Возвращает функцию награды для текущей конфигурации маятника."""
        return build_reward(self.reward, self.plant)

    def make_terminate(self) -> TerminateFn:
        """Возвращает функцию terminate для текущей стратегии."""
        return build_terminate(self.terminate)

    def make_truncate(self) -> TruncateFn:
        """Возвращает функцию truncate для текущей стратегии."""
        return build_truncate(self.truncate)

    def make_target(self) -> TargetFn:
        """Возвращает функцию target(time) для текущей стратегии цели."""
        return build_target(self.target)

    # ── Варианты конфигурации ──────────────────────────────────────────────
    def with_plant(self, **overrides: Any) -> ExperimentConfig:
        """Возвращает копию с изменёнными параметрами растения.

        Параметры растения мутируемы, поэтому копия обязательна: общие
        конфигурации не должны расходиться между запусками.
        """
        plant = self.plant.copy()
        for key, value in overrides.items():
            if not hasattr(plant, key):
                raise AttributeError(f"У PlantConfig нет поля {key!r}")
            setattr(plant, key, value)
        return dataclasses.replace(self, plant=plant)

    # ── Сериализация ───────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """Возвращает детерминированный словарь всего эксперимента."""
        return {
            "name": self.name,
            "reward": self.reward,
            "terminate": self.terminate,
            "truncate": self.truncate,
            "target": self.target,
            "n_envs": self.n_envs,
            "plant": self.plant.to_dict(),
            "sensor": self.sensor.to_dict(),
            "controller": self.controller.to_dict(),
            "ppo": dataclasses.asdict(self.ppo),
        }


def build_env(
    experiment: ExperimentConfig,
    *,
    max_episode_steps: int | None = None,
) -> PendulumEnv:
    """Создаёт одиночную Gymnasium-среду по описанию эксперимента.

    Импорт Gymnasium-слоя отложен: конфигурацию можно читать (и
    сериализовать) без инициализации среды.
    """
    from packages.simulation.env.env import PendulumEnv as _PendulumEnv

    extra: dict[str, Any] = {}
    if max_episode_steps is not None:
        extra["max_episode_steps"] = max_episode_steps

    return _PendulumEnv(
        experiment.plant,
        experiment.sensor,
        experiment.make_reward(),
        experiment.make_terminate(),
        experiment.controller,
        experiment.make_target(),
        experiment.make_truncate(),
        **extra,
    )


def make_env_factory(experiment: ExperimentConfig) -> EnvFactory:
    """Возвращает фабрику сред для пула параллельных окружений.

    Каждый вызов создаёт новую среду — именно этого ждут ``SubprocVecEnv``
    и ``DummyVecEnv``. Тренер получает фабрику, а не готовый пул, поэтому
    слой контроллеров не зависит от устройства пула.
    """
    max_episode_steps = experiment.ppo.max_episode_steps

    def factory() -> gym.Env:
        return build_env(experiment, max_episode_steps=max_episode_steps)

    return factory


__all__ = [
    "EnvFactory",
    "ExperimentConfig",
    "TargetFn",
    "TargetSpec",
    "build_env",
    "build_target",
    "make_env_factory",
    "upright_target",
]

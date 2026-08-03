from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from controller import Controller
from datatypes import ControllerConfig
from loggers import Logger

from packages.controllers.PPO.mode_config import PPOConfig
from packages.simulation.CO import (
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorConfig,
)
from packages.simulation.ENV.env import PendulumEnv


class PPOController(Controller):
    """
    PPO-агент, обёрнутый в интерфейс базового класса Controller.

    Использует готовую реализацию PPO из Stable-Baselines3,
    но предоставляет единый API через наследование от ``Controller``:

    - ``get_action()`` — детерминированное действие обученной политики
    - ``train()`` — запуск обучения PPO на ``PendulumEnv``
    - ``save()`` / ``load()`` — сохранение/загрузка модели SB3
    - ``reset()`` — сброс состояния

    Parameters
    ----------
    ppo_config : PPOConfig
        Гиперпараметры PPO-алгоритма.
    controller_config : ControllerConfig
        Конфигурация базового контроллера (такт, макс. сила, фильтры).
    plant_config : PlantConfig
        Конфигурация физической модели (для создания среды).
    sensor_config : SensorConfig
        Конфигурация сенсоров (для создания среды).
    noise : NoiseForce
        Параметры внешнего возмущения.
    target_state : np.ndarray
        Целевой вектор состояния.
    model : SB3_PPO | None
        Готовая SB3-модель (если уже загружена из файла).
    """

    def __init__(
        self,
        ppo_config: PPOConfig,
        controller_config: ControllerConfig,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        target_state: np.ndarray,
        model: SB3_PPO | None = None,
    ) -> None:
        super().__init__(controller_config)
        self.name = "PPO"

        self._ppo_config = ppo_config
        self._plant_config = plant_config
        self._sensor_config = sensor_config
        self._noise = noise
        self._target_state = target_state

        # SB3-модель (создаётся в train() или загружается из файла)
        self._model: SB3_PPO | None = model

        # Внутренняя среда для get_action
        self._env: PendulumEnv | None = None

    # ── Закон управления (абстрактный метод Controller) ──────────────────

    def get_control(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
        """
        Получить действие от обученной PPO-политики.

        Вектор наблюдения для агента формируется как конкатенация
        текущего состояния и целевого состояния (длина 12).

        Parameters
        ----------
        s_clean : np.ndarray
            Отфильтрованный вектор состояния ``(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)``.
        target_state : np.ndarray
            Целевой вектор состояния ``(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)``.

        Returns
        -------
        float
            Управляющая сила (Н).
        """
        if self._model is None:
            raise RuntimeError("PPO-модель не загружена. Вызовите train() или load().")

        obs = np.concat([np.asarray(s_clean, dtype=np.float64),
                         np.asarray(target_state, dtype=np.float64)])
        action, _ = self._model.predict(obs, deterministic=True)
        return float(action.item())

    # ── Обучение ─────────────────────────────────────────────────────────

    def train(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        target_state: np.ndarray,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
        episode_max_time: float = 150.0,
        epochs: int = 1000,
        episodes_per_epoch: int = 100,
        logger: Optional[Logger] = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Запустить обучение PPO-агента.

        Параметры, специфичные для PPO, берутся из ``PPOConfig``,
        переданного в конструктор. Остальные параметры игнорируются
        для совместимости с общим API ``Controller.train()``.

        Notes
        -----
        Для тонкой настройки используйте ``PPOConfig`` напрямую.
        """
        ppo_cfg = self._ppo_config

        # ── Фабрика среды ───────────────────────────────────────────────
        def make_env() -> PendulumEnv:
            return PendulumEnv(
                plant_config=plant_config,
                sensor_config=sensor_config,
                controller=self,
                noise_force=noise,
                target_state=target_state,
                max_force=self._max_force,
                max_episode_steps=ppo_cfg.max_episode_steps,
            )

        vec_env = DummyVecEnv([make_env])

        # ── Callback'и ──────────────────────────────────────────────────
        ckpt_dir = Path("checkpoints") / self.name.lower()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=max(50_000, ppo_cfg.n_steps),
            save_path=str(ckpt_dir),
            name_prefix="ppo",
        )
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=5000.0,
            verbose=1,
        )
        eval_env = DummyVecEnv([make_env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            eval_freq=max(20_000, ppo_cfg.n_steps),
            deterministic=True,
            callback_after_eval=stop_callback,
        )

        # ── Создание PPO-модели ─────────────────────────────────────────
        self._model = SB3_PPO(
            ppo_cfg.policy,
            vec_env,
            learning_rate=ppo_cfg.learning_rate,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            n_epochs=ppo_cfg.n_epochs,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            clip_range=ppo_cfg.clip_range,
            ent_coef=ppo_cfg.ent_coef,
            max_grad_norm=ppo_cfg.max_grad_norm,
            verbose=1,
        )

        # ── Запуск обучения ─────────────────────────────────────────────
        self._model.learn(
            total_timesteps=ppo_cfg.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
        )

    # ── Сохранение / загрузка ───────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Сохранить PPO-модель SB3 в файл.

        Parameters
        ----------
        path : str | Path
            Путь к файлу (.zip).
        """
        if self._model is None:
            raise RuntimeError("Нечего сохранять — модель не обучена.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        ppo_config: PPOConfig,
        controller_config: ControllerConfig,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        target_state: np.ndarray,
    ) -> PPOController:
        """
        Создать PPOController с предобученной моделью.

        Parameters
        ----------
        path : str | Path
            Путь к файлу SB3-модели (.zip).
        ppo_config : PPOConfig
            Гиперпараметры PPO.
        controller_config : ControllerConfig
            Конфигурация базового контроллера.
        plant_config : PlantConfig
            Конфигурация физической модели.
        sensor_config : SensorConfig
            Конфигурация сенсоров.
        noise : NoiseForce
            Параметры внешнего возмущения.
        target_state : np.ndarray
            Целевой вектор состояния.

        Returns
        -------
        PPOController
            Экземпляр с загруженной моделью.
        """
        model = SB3_PPO.load(str(path))
        return cls(
            ppo_config=ppo_config,
            controller_config=controller_config,
            plant_config=plant_config,
            sensor_config=sensor_config,
            noise=noise,
            target_state=target_state,
            model=model,
        )

    def load(self, path: str | Path) -> None:
        """
        Загрузить PPO-модель SB3 из файла.

        Parameters
        ----------
        path : str | Path
            Путь к файлу (.zip).
        """
        self._model = SB3_PPO.load(str(path))

    # ── Сброс ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Сбросить состояние контроллера."""
        super().reset()
        # PPO не хранит внутреннего состояния,
        # но сбрасываем среду если она была создана
        if self._env is not None:
            self._env.close()
            self._env = None
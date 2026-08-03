from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np

from packages.simulation.CO.datatypes import NoiseForce, PlantConfig, SensorConfig
from packages.simulation.CO.pendulum import ObjectOfControl
from packages.simulation.CO.sensor import SensorBlock


class PendulumEnv(gym.Env):
    """
    Gym-совместимая обёртка для симуляции маятника на тележке.

    Инкапсулирует физическую модель (ObjectOfControl) и блок сенсоров
    (SensorBlock), предоставляя стандартный интерфейс gym.Env для
    обучения RL-агентов.

    Наблюдение (observation):
        Вектор состояния ``(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)``,
        полученный с сенсоров (с шумом и квантованием).

    Действие (action):
        Непрерывное скалярное значение — сила (Н), приложенная к тележке.
        Клиппируется в диапазон ``[-max_force, +max_force]``.

    Parameters
    ----------
    plant_config : PlantConfig
        Конфигурация физических параметров объекта управления.
    sensor_config : SensorConfig
        Конфигурация датчиков (шум, квантование).
    noise_force : NoiseForce
        Параметры внешнего возмущающего воздействия (белый шум).
    target_state : np.ndarray | None
        Целевой вектор состояния. По умолчанию — нулевой (6,).
    max_force : float
        Максимальная сила, ограничивающая действие агента (Н).
    max_episode_steps : int
        Максимальное число шагов в эпизоде (для ``truncated``).
    reward_function : Callable | None
        Пользовательская функция награды.
        Сигнатура: ``reward_fn(target: np.ndarray, measured: np.ndarray) -> float``.
        Если ``None`` — используется штраф за отклонение от целевого состояния.
    dt_control : float
        Такт управления (с). По умолчанию 0.005 (200 Гц).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise_force: NoiseForce | None = None,
        target_state: np.ndarray | None = None,
        max_force: float = 30.0,
        max_episode_steps: int = 1000, # TODO Заменить на секунды
        reward_function: Callable[[np.ndarray, np.ndarray], float] | None = None,
        dt_control: float = 0.005,
    ) -> None:
        super().__init__()

        # ── Конфигурации ────────────────────────────────────────────────
        self._plant_config = plant_config
        self._sensor_config = sensor_config
        self._noise_force = noise_force or NoiseForce(mean=0.0, std=0.0)
        self._target_state = (
            target_state if target_state is not None else np.zeros(6, dtype=np.float64)
        )
        self._max_force = float(max_force)
        self._max_episode_steps = int(max_episode_steps)
        self._reward_function = reward_function
        self._dt_control = float(dt_control)

        # ── Компоненты симуляции ────────────────────────────────────────
        self._plant: ObjectOfControl = ObjectOfControl(self._plant_config)
        self._sensor = SensorBlock(self._sensor_config)


        # ── Пространства Gym ────────────────────────────────────────────
        # Наблюдение: (x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float64)

        # Действие: сила на тележке (Н)
        self.action_space = gym.spaces.Box(
            low=-self._max_force,
            high=self._max_force,
            shape=(1,),
            dtype=np.float64,
        )

        # ── Счётчики ────────────────────────────────────────────────────
        self._current_step: int = 0
        self._prev_force: float = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Основные методы Gym
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Сбросить среду в начальное состояние.

        Parameters
        ----------
        seed : int | None
            Seed для генератора случайных чисел (воспроизводимость).
        options : dict[str, Any] | None
            Дополнительные опции (не используются).

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            ``(observation, info)``:
            - ``observation`` — начальный вектор состояния (6,)
            - ``info`` — пустой словарь (зарезервировано)
        """
        super().reset(seed=seed)

        # Пересоздаём объекты симуляции
        self._plant.reset()

        self._current_step = 0
        self._prev_force = 0.0

        obs = self._get_observation()
        return obs, {}

    def step(
        self, action: np.ndarray | float
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Выполнить один шаг симуляции.

        Алгоритм:
        1. Клиппировать действие в диапазон ``[-max_force, +max_force]``.
        2. Выполнить микрошаги физики с новым управлением.
        3. Получить телеметрию с сенсоров.
        4. Вычислить награду.
        5. Проверить терминальные условия.

        Parameters
        ----------
        action : np.ndarray | float
            Управляющая сила (Н). Может быть скаляром или массивом (1,).

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict[str, Any]]
            ``(observation, reward, terminated, truncated, info)``:
            - ``observation`` — новый вектор состояния (6,)
            - ``reward`` — награда за шаг
            - ``terminated`` — ``True`` если эпизод завершён аварийно
            - ``truncated`` — ``True`` если превышен лимит шагов
            - ``info`` — дополнительная диагностика
        """
        if self._plant is None or self._sensor is None:
            raise RuntimeError("Среда не инициализирована. Вызовите reset() перед step().")

        # ── 1. Клиппирование действия ───────────────────────────────────
        if isinstance(action, np.ndarray):
            action = float(action.item())

        # ── 2. Шаг физики ───────────────────────────────────────────────

        steps_per_control = int(self._dt_control / self._plant._dt)
        steps_per_control_compute = int(0.2*steps_per_control)             # TODO Добавить в настроки коефициент задержки
        steps_per_control_action = steps_per_control - steps_per_control_compute

        for _ in range(steps_per_control_compute):
            self._plant.update_physics(self._prev_force, self._noise_force)
        for _ in range(steps_per_control_action):
            self._plant.update_physics(action, self._noise_force)

        # ── 3. Телеметрия ───────────────────────────────────────────────
        obs = self._get_observation()

        # ── 4. Награда ──────────────────────────────────────────────────
        reward = self._compute_reward(obs)

        # ── 5. Терминальные условия ─────────────────────────────────────
        self._current_step += 1
        self._prev_force = action
        terminated = self._check_terminated()
        truncated = self._current_step >= self._max_episode_steps

        info: dict[str, Any] = {
            "step": self._current_step,
            "force": action,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """
        Рендеринг среды (заглушка).

        Parameters
        ----------
        mode : str
            Режим рендеринга (``"human"`` или ``"rgb_array"``).
        """
        pass

    def close(self) -> None:
        """Освободить ресурсы среды."""
        self._plant = None
        self._sensor = None

    # ──────────────────────────────────────────────────────────────────────
    # Внутренние методы
    # ──────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Получить текущее наблюдение с сенсоров.

        Returns
        -------
        np.ndarray
            Вектор ``(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)`` с шумом и квантованием.
        """
        if self._plant is None or self._sensor is None:
            return np.zeros(6, dtype=np.float64)
        return self._sensor.get_telemetry(self._plant.q, self._plant.dq)

    def _compute_reward(self, obs: np.ndarray) -> float:
        """
        Вычислить награду за текущий шаг.

        Если задана пользовательская ``reward_function`` — использует её.
        Иначе — отрицательный квадрат евклидова расстояния до целевого состояния.

        Parameters
        ----------
        obs : np.ndarray
            Текущее наблюдение (6,).

        Returns
        -------
        float
            Значение награды.
        """
        if self._reward_function is not None:
            return self._reward_function(self._target_state, obs)

        # По умолчанию: штраф за отклонение от цели
        diff = obs - self._target_state
        return 1

    def _check_terminated(self) -> bool:
        """
        Проверить, завершён ли эпизод аварийно.

        Условия:
        - Отклонение маятника от вертикали более чем на 40°
        - Выход тележки за пределы ``[-3, 3]`` м

        Returns
        -------
        bool
            ``True`` если эпизод должен быть завершён.
        """
        if self._plant is None:
            return False

        q = self._plant.q
        angle = q[1]
        deviation = abs(angle - np.pi)
        return deviation > np.radians(40)

    # ──────────────────────────────────────────────────────────────────────
    # Свойства для доступа к внутренним компонентам
    # ──────────────────────────────────────────────────────────────────────

    @property
    def plant(self) -> ObjectOfControl | None:
        """Объект управления (физическая модель)."""
        return self._plant

    @property
    def sensor(self) -> SensorBlock | None:
        """Блок сенсоров."""
        return self._sensor

    @property
    def target_state(self) -> np.ndarray:
        """Целевой вектор состояния."""
        return self._target_state

    @target_state.setter
    def target_state(self, value: np.ndarray) -> None:
        self._target_state = np.asarray(value, dtype=np.float64)
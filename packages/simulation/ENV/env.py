from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from packages.simulation.CO.datatypes import ControllerConfig, PlantConfig, SensorConfig
from packages.simulation.CO.pendulum import ObjectOfControl


class PendulumEnv(gym.Env):
    DEFAULT_MAX_EPISODE_STEPS = 2_000

    def __init__(
        self,
        plant_config,
        sensor_config,
        cost_function,
        terminate_condition:Callable[[np.ndarray, np.ndarray], bool],
        controller_config:ControllerConfig,
        target: Callable[[float], np.ndarray],
        truncated_condition:Callable[[np.ndarray, np.ndarray], bool],
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    ) -> None:
        super().__init__()
        self._t = 0
        self._truncated_condition = truncated_condition
        self._target:Callable[[float], np.ndarray] = target
        self._plant_config:PlantConfig = plant_config
        self._sensor_config:SensorConfig = sensor_config

        self._plant:ObjectOfControl = ObjectOfControl(plant_config, sensor_config)

        self._old_action: float = 0

        self._cost_function:Callable = cost_function
        self._terminate_condition:Callable[[np.ndarray, np.ndarray], bool] = terminate_condition

        self._controller_dt = controller_config.dt
        self._max_episode_steps = int(max_episode_steps)
        if self._max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self._episode_steps = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(14,),
            dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-controller_config.max_force,  
            high=controller_config.max_force,
            shape=(1,),
            dtype=np.float64
        )


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
 
        super().reset()

        self._t = 0
        self._episode_steps = 0

        self._plant.reset()
        obs = np.concatenate([self._plant.get_telemetry(), self._target(self._t)])
        return (obs,{})


    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    
        updates_total = int(self._controller_dt/self._plant._dt)
        prev_a_upds = int(0.2*updates_total)
        new_a_upds = updates_total - prev_a_upds

        # обновить физику на протяжении 20% от одного такта контроллера с старой силой
        self._plant.update_physics(action.item(), prev_a_upds) 
        # обновить физику на протяжении 80% от одного такта контроллера с новой силой
        self._plant.update_physics(action.item(), new_a_upds)
        # вычислить награду

        observation = self._plant.get_telemetry()
        target_t = self._target(self._t)
        obs = np.concatenate([observation, target_t])

        # Награда считается по состоянию и цели; текущая энергетическая
        # reward-функция не зависит от управляющей силы.
        reward = self._cost_function(observation, target_t)

        # проверить на терминальность
        terminate_flag = self._terminate_condition(observation, target_t)
        truncated_flag = self._truncated_condition(observation, target_t)
        self._t += self._controller_dt
        self._episode_steps += 1
        truncated_flag = truncated_flag or (
            self._episode_steps >= self._max_episode_steps
        )
        # вернуть значения  
        return obs, reward, terminate_flag, truncated_flag, {}
    

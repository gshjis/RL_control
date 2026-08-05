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
    def __init__(
        self,
        plant_config,
        sensor_config,
        cost_function,
        terminate_condition:Callable[[np.ndarray], bool],
        controller_config:ControllerConfig,
        target: Callable[[float], np.ndarray]
    ) -> None:
        super().__init__()
        self._t = 0
        self._target:Callable[[float], np.ndarray] = target
        self._plant_config:PlantConfig = plant_config
        self._sensor_config:SensorConfig = sensor_config

        self._plant:ObjectOfControl = ObjectOfControl(plant_config, sensor_config)

        self._old_action: float = 0

        self._cost_function:Callable = cost_function
        self._terminate_condition:Callable[[np.ndarray], bool] = terminate_condition

        self._controller_dt = controller_config.dt

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(6,),
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

        self._plant.reset()
        r = np.concatenate(self._plant.get_clean_state())
        return (r,{})


    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    
        updates_total = int(self._controller_dt/self._plant._dt)
        prev_a_upds = int(0.2*updates_total)
        new_a_upds = updates_total - prev_a_upds

        # обновить физику на протяжении 20% от одного такта контроллера с старой силой
        self._plant.update_physics(self._old_action, prev_a_upds) # TODO второе число, количество тактов, возмущения реализуются в С++
        # обновить физику на протяжении 80% от одного такта контроллера с новой силой
        self._plant.update_physics(action.item(), new_a_upds)  # TODO второе число, количество тактов, возмущения реализуются в С++

        # вычислить награду
        observation = self._plant.get_telemetry()
        target_t = self._target(self._t)
        reward = self._cost_function(observation-target_t)

        # проверить на терминальность
        terminate_flag = self._terminate_condition(observation-target_t,)

        self._t += self._controller_dt
        # вернуть значения  
        return observation-target_t, reward, terminate_flag, False, {}
    
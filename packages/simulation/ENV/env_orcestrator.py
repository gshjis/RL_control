from collections.abc import Callable

import gymnasium as gym
import numpy as np
from datatypes import ControllerConfig
from stable_baselines3.common.vec_env import SubprocVecEnv

from .env import PendulumEnv


class EnvOrchestrator:
    def __init__(
        self,
        plant_config,
        sensor_config,
        cost_function,
        terminate_condition:Callable[[np.ndarray], bool],
        n_simulations: int,
        controller_config:ControllerConfig,
        truncated_condition:Callable[[np.ndarray], bool],        
        target: Callable[[float], np.ndarray]
    ):
        self._truncated_condition:Callable[[np.ndarray], bool] = truncated_condition
        self._target:Callable[[float], np.ndarray] = target
        self._controller_config:ControllerConfig = controller_config
        self._plant_config = plant_config
        self._sensor_config = sensor_config
        self._cost_function = cost_function
        self._terminate_condition = terminate_condition
        
        env_fns = [
            self._make_env for _ in range(n_simulations)
        ]
        
        self._env_hub = SubprocVecEnv(env_fns)

    
    def _make_env(self) -> gym.Env:
        """Фабрика для создания одной среды."""
        return PendulumEnv(
            self._plant_config,
            self._sensor_config,
            self._cost_function,
            self._terminate_condition,
            self._controller_config,
            self._target,
            self._truncated_condition
        )
    

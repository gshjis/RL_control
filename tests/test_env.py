import numpy as np
from gymnasium import spaces

from packages.controllers.base import terminate_condition, truncated_condition
from packages.simulation.co.datatypes import ControllerConfig, PlantConfig, SensorConfig
from packages.simulation.env.env import PendulumEnv


def make_env() -> PendulumEnv:
    plant_config = PlantConfig(
        init_q_stats=np.zeros((2, 3)),
        init_dq_stats=np.zeros((2, 3)),
    )
    sensor_config = SensorConfig(
        noise_std_q=(0.0, 0.0, 0.0), noise_std_dq=(0.0, 0.0, 0.0)
    )
    controller_config = ControllerConfig(dt=0.001, max_force=5.0)
    return PendulumEnv(
        plant_config,
        sensor_config,
        lambda state, target: float(np.sum(state) + np.sum(target)),
        terminate_condition,
        controller_config,
        lambda _: np.zeros(6),
        truncated_condition,
        max_episode_steps=3,
    )


def test_env_spaces_match_contract() -> None:
    env = make_env()

    assert env.observation_space.shape == (14,)
    assert env.action_space.shape == (1,)
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.high[0] == 5.0


def test_env_reset_returns_observation_and_info() -> None:
    env = make_env()

    observation, info = env.reset(seed=1)

    assert observation.shape == (14,)
    assert observation.dtype == np.float64
    assert info == {}
    env.close()


def test_env_step_returns_gymnasium_tuple() -> None:
    env = make_env()
    env.reset()

    result = env.step(np.zeros(1, dtype=np.float64))

    observation, reward, terminated, truncated, info = result
    assert observation.shape == (14,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info == {}
    env.close()


def test_env_truncates_at_episode_limit() -> None:
    env = make_env()
    env.reset()
    action = np.zeros(1, dtype=np.float64)

    _, _, _, truncated, _ = env.step(action)
    assert not truncated
    _, _, _, truncated, _ = env.step(action)
    assert not truncated
    _, _, _, truncated, _ = env.step(action)
    assert truncated
    env.close()

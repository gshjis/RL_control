import numpy as np

from packages.controllers.PPO import PPOConfig
from packages.simulation.CO import (
    ControllerConfig,
    NoiseForce,
    PlantConfig,
    SensorConfig,
)

ppo_config = PPOConfig(
    total_timesteps=3_500_000,
    eval_freq=100_000,
    n_eval_episodes=2,
    max_episode_steps=2_000,
)

PLANT_CONFIG = PlantConfig(
    M=1.0,
    m1=0.05,
    l1=0.2,
    m2=0.0,
    l2=0.0,
    g=-9.81,
    b_c=0.01,
    b_1=0.001,
    b_2=0.001,
    single_pendulum_mode=True,
    init_q_stats=np.array([
        [0.0, np.pi, 0.0],
        [0.01, 0.00, 0.01]]),
    init_dq_stats=np.array([
        [0.0, 0.0, 0.0], 
        [0.0, 0.00, 0.01]]),
    dt=0.001,
    motor_time_constant=0.01,
)

SENSOR_CONFIG = SensorConfig(
    encoder_resolution_1=4096,
    encoder_resolution_2=4096,
    cart_sensor_resolution=0.0001,
    noise_std_q=(0.0005, 0.002, 0.002),
    noise_std_dq=(0.005, 0.01, 0.01),
)

CONTROLLER_CONFIG = ControllerConfig(
    dt=0.01,
    max_force=24.0,
    has_velocity_sensors=False,
    filter_cutoff_hz=50.0,
)

NOISE = NoiseForce(mean=0.00, std=0.03)


def target(time: float) -> np.ndarray:
    return np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NoiseForce:
    mean: float = 0.0
    std: float = 0.0


@dataclass
class PlantConfig:

    M: float = 1.0
    m1: float = 0.3
    m2: float = 0.0
    l1: float = 1.0
    l2: float = 0.0
    g: float = -9.81

    b_c: float = 0.0
    b_1: float = 0.0
    b_2: float = 0.0

    single_pendulum_mode: bool = True
    motor_time_constant: float = 0.0

    init_q_stats: np.ndarray = field(default_factory=lambda: np.array([[0.0, np.pi, 0.0], [0.1, 0.1, 0.0]]))
    init_dq_stats: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]]))
    dt: float = 0.0005

    mean_f:float = 0
    std_f:float = 0.02

    L1: float = field(init=False)
    L2: float = field(init=False)
    J1: float = field(init=False)
    J2: float = field(init=False)


    def __post_init__(self) -> None:
        object.__setattr__(self, "L1", self.l1 / 2.0)
        object.__setattr__(self, "L2", self.l2 / 2.0)
        object.__setattr__(self, "J1", (1.0 / 12.0) * self.m1 * self.l1**2)
        object.__setattr__(self, "J2", (1.0 / 12.0) * self.m2 * self.l2**2)

    def init_q(self) -> np.ndarray:
        """Sample an initial position from the lower/upward 50/50 mixture.

        With the simulator's angle convention, ``theta1 = 0`` is the lower
        position and ``theta1 = pi`` is the upright position. The remaining
        coordinates retain the configured Gaussian initialization.
        """
        t = np.random.normal(self.init_q_stats[0], self.init_q_stats[1])
        t[1] = np.pi if np.random.randint(0, 2) else 0.0
        return t
    def init_dq(self) -> np.ndarray:
        t = np.random.normal(self.init_dq_stats[0], self.init_dq_stats[1])
        return t
    def to_dict(self) -> dict:
        return {
            "M": self.M,
            "m1": self.m1,
            "m2": self.m2,
            "l1": self.l1,
            "l2": self.l2,
            "L1": self.L1,
            "L2": self.L2,
            "J1": self.J1,
            "J2": self.J2,
            "g": self.g,
            "b_c": self.b_c,
            "b_1": self.b_1,
            "b_2": self.b_2,
            "single_pendulum_mode": self.single_pendulum_mode,
            "motor_time_constant": self.motor_time_constant,
            "init_q": self.init_q().tolist(),
            "init_dq": self.init_dq().tolist(),
            "dt": self.dt
        }

    def copy(self) -> PlantConfig:
        """
        Создать глубокую копию конфигурации.

        Returns
        -------
        PlantConfig
            Независимая копия со своими массивами ``init_q`` / ``init_dq``.
        """
        return PlantConfig(
            M=self.M,
            m1=self.m1,
            m2=self.m2,
            l1=self.l1,
            l2=self.l2,
            g=self.g,
            b_c=self.b_c,
            b_1=self.b_1,
            b_2=self.b_2,
            single_pendulum_mode=self.single_pendulum_mode,
            motor_time_constant=self.motor_time_constant,
            init_q_stats=self.init_q_stats,
            init_dq_stats=self.init_dq_stats,
            dt=self.dt,
        )


@dataclass
class SensorConfig:

    encoder_resolution_1: int = 4096
    encoder_resolution_2: int = 4096
    cart_sensor_resolution: float = 0.0001

    noise_std_q: list[float] | tuple[float, float, float] = (
        0.001,
        0.005,
        0.005,
    )
    noise_std_dq: list[float] | tuple[float, float, float] = (
        0.01,
        0.02,
        0.02,
    )
    seed: int | None = None
    noise_pool_size: int = 2_000_000

    differentiator_cutoff_hz: float | None = None
    filter_cutoff_hz: float = 50.0

    def to_dict(self) -> dict:
        return {
            "encoder_resolution_1": self.encoder_resolution_1,
            "encoder_resolution_2": self.encoder_resolution_2,
            "cart_sensor_resolution": self.cart_sensor_resolution,
            "noise_std_q": list(self.noise_std_q),
            "noise_std_dq": list(self.noise_std_dq),
            "seed": self.seed,
            "noise_pool_size": self.noise_pool_size,
            "differentiator_cutoff_hz": self.differentiator_cutoff_hz,
            "filter_cutoff_hz": self.filter_cutoff_hz,
        }


@dataclass
class ControllerConfig:

    dt: float = 0.005
    max_force: float = 30.0
    has_velocity_sensors: bool = False
    differentiator_cutoff_hz: float | None = None
    filter_cutoff_hz: float = 50.0

    def to_dict(self) -> dict:
        return {
            "dt": self.dt,
            "max_force": self.max_force,
            "has_velocity_sensors": self.has_velocity_sensors,
            "differentiator_cutoff_hz": self.differentiator_cutoff_hz,
            "filter_cutoff_hz": self.filter_cutoff_hz,
        }

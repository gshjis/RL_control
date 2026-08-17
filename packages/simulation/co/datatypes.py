from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NoiseForce:
    """Хранит среднее и СКО внешнего силового шума."""

    mean: float = 0.0
    std: float = 0.0


@dataclass
class PlantConfig:
    """Хранит параметры физической модели маятника."""

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

    init_q_stats: np.ndarray = field(
        default_factory=lambda: np.array([[0.0, np.pi, 0.0], [0.1, 0.1, 0.0]])
    )
    init_dq_stats: np.ndarray = field(
        default_factory=lambda: np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]])
    )
    dt: float = 0.0005

    mean_f: float = 0
    std_f: float = 0.02

    L1: float = field(init=False)
    L2: float = field(init=False)
    J1: float = field(init=False)
    J2: float = field(init=False)

    def __post_init__(self) -> None:
        """Вычисляет производные геометрические и инерционные параметры."""
        object.__setattr__(self, "L1", self.l1 / 2.0)
        object.__setattr__(self, "L2", self.l2 / 2.0)
        object.__setattr__(self, "J1", (1.0 / 12.0) * self.m1 * self.l1**2)
        object.__setattr__(self, "J2", (1.0 / 12.0) * self.m2 * self.l2**2)

    def init_q(self) -> np.ndarray:
        """Принимает настройки распределения и возвращает начальные координаты."""
        t = np.random.normal(self.init_q_stats[0], self.init_q_stats[1])
        t[1] = np.pi if np.random.randint(0, 2) else 0.0
        return t

    def init_dq(self) -> np.ndarray:
        """Возвращает начальные скорости из заданного нормального распределения."""
        t = np.random.normal(self.init_dq_stats[0], self.init_dq_stats[1])
        return t

    def to_dict(self) -> dict:
        """Возвращает конфигурацию в виде словаря сериализуемых значений."""
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
            "dt": self.dt,
        }

    def copy(self) -> PlantConfig:
        """Возвращает независимую копию конфигурации растения."""
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
            init_q_stats=self.init_q_stats.copy(),
            init_dq_stats=self.init_dq_stats.copy(),
            dt=self.dt,
        )


@dataclass
class SensorConfig:
    """Хранит параметры разрешения, шума и фильтрации датчиков."""

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
        """Возвращает конфигурацию датчиков в виде словаря."""
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
    """Хранит параметры шага, силы и фильтров контроллера."""

    dt: float = 0.005
    max_force: float = 30.0
    has_velocity_sensors: bool = False
    differentiator_cutoff_hz: float | None = None
    filter_cutoff_hz: float = 50.0

    def to_dict(self) -> dict:
        """Возвращает конфигурацию контроллера в виде словаря."""
        return {
            "dt": self.dt,
            "max_force": self.max_force,
            "has_velocity_sensors": self.has_velocity_sensors,
            "differentiator_cutoff_hz": self.differentiator_cutoff_hz,
            "filter_cutoff_hz": self.filter_cutoff_hz,
        }

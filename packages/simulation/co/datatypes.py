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

    # Вероятность инициализации угла theta в верхнем положении (theta=pi).
    p_upright: float = 0.01

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

    def init_q(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Возвращает начальные координаты.

        Без явного генератора использует глобальный ``np.random`` (устаревший
        путь, оставлен для обратной совместимости). Среда передаёт сюда
        свой ``self.np_random``, чтобы reset(seed=...) делал эпизод
        воспроизводимым.
        """
        generator = rng if rng is not None else np.random
        t = generator.normal(self.init_q_stats[0], self.init_q_stats[1])
        t[1] = np.pi if generator.random() <= self.p_upright else 0.0
        return t

    def init_dq(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Возвращает начальные скорости (см. init_q про генератор)."""
        generator = rng if rng is not None else np.random
        t = generator.normal(self.init_dq_stats[0], self.init_dq_stats[1])
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
            # Сериализуются параметры распределений, а не разовые выборки:
            # to_dict() должен быть детерминированным (именно он попадает в
            # артефакты эксперимента рядом с чекпоинтом).
            "init_q_stats": self.init_q_stats.tolist(),
            "init_dq_stats": self.init_dq_stats.tolist(),
            "p_upright": self.p_upright,
            "mean_f": self.mean_f,
            "std_f": self.std_f,
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
            p_upright=self.p_upright,
            mean_f=self.mean_f,
            std_f=self.std_f,
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
    # Пул шума хранится в памяти (3 канала × 8 байт на сэмпл):
    # 100k сэмплов ≈ 5 МБ на блок датчиков и хватает на 100 с симуляции
    # при dt=1 мс. Старое значение 2_000_000 съедало ~96 МБ на среду,
    # что с SubprocVecEnv (8 процессов) выливалось в гигабайты RSS.
    noise_pool_size: int = 100_000

    # Отсечки Гц: None у дифференциатора означает чистое дифференцирование
    # без сглаживания (шум производной не подавляется) — осознанно не
    # используем, по умолчанию стоит рабочее значение 40 Гц.
    differentiator_cutoff_hz: float | None = 40.0
    filter_cutoff_hz: float = 50.0

    # True — «идеальный наблюдатель» (дефолт): точные углы и скорости из
    # физики, без квантования и шума — воспроизводимо и проще в обучении.
    # False — честные датчики: квантование энкодеров + шум + оценка
    # скоростей дифференцированием; обучение переносится на реальную
    # аппаратуру, но меняет цифры наград (старые чекпоинты несопоставимы).
    exact_mode: bool = True

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
            "exact_mode": self.exact_mode,
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

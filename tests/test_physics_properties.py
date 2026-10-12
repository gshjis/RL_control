"""Аналитические проверки физического ядра (C++/pybind11).

Это «золотой» слой рефакторинга: он не содержит захардкоженных траекторий,
но фиксирует физические свойства, которые обязаны сохраняться при любых
правках кода — незыблемость верхнего равновесия, сохранение энергии в
отсутствие рассеивания, детерминизм и формат телеметрии.
"""

from __future__ import annotations

import numpy as np

from packages.simulation.co import ObjectOfControl, PlantConfig, SensorConfig
from packages.simulation.co.layout import (
    CLEAN_IDX_DTH1,
    CLEAN_IDX_DTH2,
    CLEAN_IDX_T1,
    CLEAN_IDX_T2,
    TELEMETRY_DIM,
)


def _plant(**overrides: object) -> PlantConfig:
    """Возвращает конфигурацию «идеального» маятника без рассеивания."""
    defaults: dict[str, object] = {
        "M": 1_000.0,  # почти неподвижная тележка: изолируем маятник
        "m1": 0.05,
        "l1": 0.2,
        "m2": 0.0,
        "l2": 0.0,
        "g": -9.81,
        "b_c": 0.0,
        "b_1": 0.0,
        "b_2": 0.0,
        "single_pendulum_mode": True,
        "motor_time_constant": 0.0,
        "mean_f": 0.0,
        "std_f": 0.0,  # без шума силы — состояние детерминировано
        "p_upright": 1.0,  # всегда старт из верхнего положения theta=pi
        "init_q_stats": np.array([[0.0, np.pi, 0.0], [0.0, 0.0, 0.0]]),
        "init_dq_stats": np.zeros((2, 3)),
        "dt": 1e-3,
    }
    defaults.update(overrides)
    return PlantConfig(**defaults)  # type: ignore[arg-type]


def _sensor() -> SensorConfig:
    """Возвращает конфигурацию датчиков без шума (сейчас отключён в C++)."""
    return SensorConfig(
        noise_std_q=(0.0, 0.0, 0.0),
        noise_std_dq=(0.0, 0.0, 0.0),
        seed=0,
    )


def _mechanical_energy(clean_state: np.ndarray, plant: PlantConfig) -> float:
    """Полная механическая энергия тележки с одно-звенным маятником."""
    _, theta1, _, dx, dtheta1, _ = clean_state

    mass = plant.m1
    half_length = plant.L1
    inertia = plant.J1
    gravity = abs(plant.g)

    # Кинетическая энергия тележки и маятника (маятник — стержень,
    # центр масс на l1/2, угол отсчитан от нижнего положения theta=0).
    t_cart = 0.5 * plant.M * dx**2
    t_rod_com = (
        0.5
        * mass
        * (
            dx**2
            + 2.0 * dx * half_length * dtheta1 * np.cos(theta1)
            + (half_length * dtheta1) ** 2
        )
    )
    t_spin = 0.5 * inertia * dtheta1**2

    # Потенциальная энергия: ноль в нижней точке (theta=0).
    potential = mass * gravity * half_length * (1.0 - np.cos(theta1))

    return float(t_cart + t_rod_com + t_spin + potential)


def test_upright_equilibrium_is_stationary() -> None:
    """Точка theta=pi при нулевой скорости и нулевой силе неподвижна."""
    pendulum = ObjectOfControl(_plant(), _sensor())

    for _ in range(200):
        pendulum.update_physics(0.0, 1)

    clean_state = pendulum.get_clean_state()
    assert abs(clean_state[CLEAN_IDX_T1] - np.pi) < 1e-9
    assert abs(clean_state[CLEAN_IDX_DTH1]) < 1e-9


def test_energy_is_conserved_without_dissipation() -> None:
    """Энергия сохраняется при нулевых трении, шуме и управляющей силе."""
    plant = _plant(init_dq_stats=np.array([[0.0, 0.05, 0.0], [0.0, 0.0, 0.0]]))
    pendulum = ObjectOfControl(plant, _sensor())

    steps = 2_000
    energies = np.empty(steps + 1, dtype=np.float64)
    energies[0] = _mechanical_energy(pendulum.get_clean_state(), plant)

    for index in range(steps):
        pendulum.update_physics(0.0, 1)
        energies[index + 1] = _mechanical_energy(pendulum.get_clean_state(), plant)

    scale = max(abs(energies[0]), 1e-12)
    drift = float(np.max(np.abs(energies - energies[0])) / scale)
    assert drift < 1e-6, f"относительное отклонение энергии: {drift:.3e}"


def test_physics_is_deterministic_without_force_noise() -> None:
    """Два одинаковых объекта дают побитово одинаковые траектории."""
    plant = _plant(init_dq_stats=np.array([[0.0, 0.05, 0.0], [0.0, 0.0, 0.0]]))

    states = []
    for _ in range(2):
        pendulum = ObjectOfControl(plant.copy(), _sensor())
        trajectory = []
        for _ in range(100):
            pendulum.update_physics(0.0, 1)
            trajectory.append(pendulum.get_clean_state())
        states.append(np.asarray(trajectory))

    np.testing.assert_array_equal(states[0], states[1])


def test_force_noise_changes_trajectory() -> None:
    """Шум силы (std_f > 0) реально попадает в траекторию."""
    plant = _plant(
        std_f=0.05,
        init_dq_stats=np.array([[0.0, 0.05, 0.0], [0.0, 0.0, 0.0]]),
    )

    trajectories = []
    for _ in range(2):
        pendulum = ObjectOfControl(plant.copy(), _sensor())
        for _ in range(100):
            pendulum.update_physics(0.0, 1)
        trajectories.append(pendulum.get_clean_state())

    assert not np.array_equal(trajectories[0], trajectories[1])


def test_single_pendulum_mode_locks_second_link() -> None:
    """В режиме одного маятника theta2 и её скорость всегда равны нулю."""
    plant = _plant(
        p_upright=0.0,
        init_q_stats=np.array([[0.0, 0.0, 0.7], [0.0, 0.0, 0.0]]),
    )
    pendulum = ObjectOfControl(plant, _sensor())

    for _ in range(50):
        pendulum.update_physics(0.0, 1)
        clean_state = pendulum.get_clean_state()
        assert clean_state[CLEAN_IDX_T2] == 0.0
        assert clean_state[CLEAN_IDX_DTH2] == 0.0


def test_telemetry_layout_matches_documented_contract() -> None:
    """Телеметрия — это [x, cos t1, sin t1, cos t2, sin t2, dx, dt1, dt2]."""
    plant = _plant(
        p_upright=0.0,
        init_q_stats=np.array([[0.05, 0.3, 0.0], [0.0, 0.0, 0.0]]),
        init_dq_stats=np.array([[0.01, 0.02, 0.03], [0.0, 0.0, 0.0]]),
    )
    pendulum = ObjectOfControl(plant, _sensor())
    clean_state = pendulum.get_clean_state()
    telemetry = pendulum.get_telemetry()

    assert telemetry.shape == (TELEMETRY_DIM,)
    x, theta1, theta2, dx, dtheta1, dtheta2 = clean_state

    expected = np.array(
        [
            x,
            np.cos(theta1),
            np.sin(theta1),
            np.cos(theta2),
            np.sin(theta2),
            dx,
            dtheta1,
            dtheta2,
        ]
    )
    np.testing.assert_allclose(telemetry, expected, rtol=0.0, atol=1e-12)

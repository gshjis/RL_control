"""Фабрики функций стоимости и награды."""

from collections.abc import Callable

import numpy as np

from packages.simulation.co.datatypes import PlantConfig
from packages.simulation.co.layout import IDX_COS_T1, IDX_DTH1


def make_energy_reward(
    plant_config: PlantConfig,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Принимает конфигурацию маятника и возвращает reward.

    Обновлено под требование эксперимента: reward = -|E_full - U_max|,
    где U_max достигается при верхнем положении (theta=pi).
    """
    m1 = float(plant_config.m1)
    half_length = float(plant_config.L1)
    inertia = float(plant_config.J1)
    gravity = abs(float(plant_config.g))
    inertia_about_pivot = inertia + m1 * half_length * half_length

    def reward(state: np.ndarray, target: np.ndarray) -> float:
        """Принимает телеметрию и цель, возвращает reward."""
        del target
        # Награда считается только по первому звену (режим одного маятника);
        # индексы берутся из packages.simulation.co.layout.
        cos_theta1 = float(state[IDX_COS_T1])
        dtheta1 = float(state[IDX_DTH1])
        kinetic_energy = 0.5 * inertia_about_pivot * dtheta1 * dtheta1
        potential_energy = m1 * gravity * half_length * (1.0 - cos_theta1)

        # Максимальная потенциальная энергия при theta=pi => cos(pi)=-1:
        # U_max = m g (L/2) (1 - (-1)) = m g (L/2) * 2
        u_max = 2.0 * m1 * gravity * half_length

        e_full = kinetic_energy + potential_energy
        return float(-abs(e_full - u_max))

    return reward

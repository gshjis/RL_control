"""Запускатель физики: PhysicsRunner."""

from __future__ import annotations

from packages.simulation.CO import ObjectOfControl, NoiseForce



class PhysicsRunner:
    """Инкапсулирует логику обновления физики.

    Принимает объект управления (plant) и выполняет шаги update_physics.
    """

    def __init__(self, plant: ObjectOfControl) -> None:
        self.plant = plant

    def step(self, force: float, noise: NoiseForce, dt: float, steps: int = 1) -> None:
        for _ in range(max(1, steps)):
            self.plant.update_physics(force, noise)


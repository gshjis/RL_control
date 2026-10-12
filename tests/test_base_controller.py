"""Проверки базового интерфейса контроллера."""

from __future__ import annotations

import numpy as np

from packages.controllers.base import Controller
from packages.simulation.co.datatypes import ControllerConfig


class DummyController(Controller):
    """Минимальный контроллер для проверки базового интерфейса."""

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Принимает наблюдение, возвращает нулевое действие."""
        del observation
        return np.zeros(1, dtype=np.float64)


def test_controller_stores_configuration() -> None:
    controller = DummyController(ControllerConfig(dt=0.02, max_force=12.0))

    assert controller.controller_config.dt == 0.02
    # Исторический прокси-доступ к полям конфигурации сохранён.
    assert controller.dt == 0.02
    assert controller.name == "DummyController"


def test_controller_str_and_repr_mention_name() -> None:
    controller = DummyController(ControllerConfig())

    assert str(controller) == "DummyController"
    assert "DummyController" in repr(controller)


def test_controller_close_is_a_no_op_by_default() -> None:
    controller = DummyController(ControllerConfig())

    controller.close()

    np.testing.assert_allclose(controller.act(np.zeros(4)), [0.0])

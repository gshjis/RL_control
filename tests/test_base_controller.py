import numpy as np

from packages.controllers.base import Controller
from packages.simulation.co.datatypes import ControllerConfig


class DummyController(Controller):
    """Минимальный контроллер для проверки базового интерфейса."""

    def action(self, state_target: np.ndarray) -> np.ndarray:
        """Возвращает нулевое действие."""
        self._last_control_action = 1.5
        return np.zeros(1, dtype=np.float64)


def test_controller_stores_configuration() -> None:
    controller = DummyController(ControllerConfig(dt=0.02, max_force=12.0))

    assert controller.dt == 0.02
    assert controller.last_control_action == 0.0


def test_controller_reset_clears_last_action() -> None:
    controller = DummyController(ControllerConfig())
    controller.action(np.zeros(1))

    assert controller.last_control_action == 1.5
    controller.reset()
    assert controller.last_control_action == 0.0

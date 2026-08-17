"""Заглушки оптимизаторов коэффициентов PID."""

from __future__ import annotations

from typing import Any


class Zigler_Nikols:
    """Заглушка оптимизатора по методу Циглера—Николса."""

    def __init__(self, logger: Any = None) -> None:
        """Принимает необязательный логгер и сохраняет его."""
        self.logger = logger

    def optimize(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Принимает параметры оптимизации, но пока не выполняет расчёт."""
        raise NotImplementedError("Оптимизатор Циглера—Николса ещё не реализован.")


class Genetic_PID_AngleOnly:
    """Заглушка генетического оптимизатора угловых PID-коэффициентов."""

    def __init__(self, logger: Any = None) -> None:
        """Принимает необязательный логгер и сохраняет его."""
        self._logger = logger

    def optimize(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Принимает параметры оптимизации, но пока не выполняет расчёт."""
        raise NotImplementedError("Генетический PID-оптимизатор ещё не реализован.")

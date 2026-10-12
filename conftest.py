"""Общие настройки pytest для проекта RL_control.

Часть тестов требует собранного C++-расширения ``co_cpp`` (см. ``make build``).
Если расширения нет, такие тесты пропускаются с понятным сообщением, а не
падают загадочным ``ModuleNotFoundError`` на этапе сбора.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest

# Тесты, которым нужно собранное C++-ядро физики/датчиков.
CPP_DEPENDENT_TESTS = frozenset(
    {
        "test_pendulum.py",
        "test_env.py",
        "test_physics_properties.py",
    }
)


def _cpp_extension_available() -> bool:
    """Возвращает признак доступности собранного расширения co_cpp."""
    try:
        return importlib.util.find_spec("packages.simulation.co.co_cpp") is not None
    except (ImportError, ValueError):
        return False


CPP_AVAILABLE = _cpp_extension_available()


def pytest_configure(config: pytest.Config) -> None:
    """Регистрирует маркер, отмечающий тесты, зависящие от C++."""
    config.addinivalue_line(
        "markers",
        "requires_cpp: тесту нужно собранное расширение co_cpp (make build)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Пропускает C++-зависимые тесты, когда расширение не собрано."""
    if CPP_AVAILABLE:
        return

    skip_cpp = pytest.mark.skip(
        reason=(
            "расширение co_cpp не собрано — выполните 'make build' "
            "(или 'make test', которая собирает ядро автоматически)"
        )
    )
    for item in items:
        if item.path is not None and item.path.name in CPP_DEPENDENT_TESTS:
            item.add_marker(skip_cpp)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """Не даёт pytest'у упасть на импорте C++-зависимых модулей сбора."""
    del config
    if CPP_AVAILABLE:
        return False

    if collection_path.name in CPP_DEPENDENT_TESTS:
        warnings.warn(
            f"{collection_path.name} пропущен: расширение co_cpp не собрано "
            "(make build)",
            stacklevel=1,
        )
        return True
    return False


# Прямой экспорт для тестов, которым нужна проверка доступности.
__all__ = ["CPP_AVAILABLE"]

"""Запуск GUI-симуляции — обёртка над ``python -m packages gui``."""

from __future__ import annotations

import sys

from packages.cli import main as cli_main


def main() -> None:
    """Запускает окно симуляции, передавая аргументы дальше в CLI."""
    raise SystemExit(cli_main(["gui", *sys.argv[1:]]))


if __name__ == "__main__":
    main()

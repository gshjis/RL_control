"""Обучение PPO-агента — обёртка над ``python -m packages train``."""

from __future__ import annotations

import sys

from packages.cli import main as cli_main


def main() -> None:
    """Запускает обучение, передавая аргументы дальше в CLI."""
    raise SystemExit(cli_main(["train", *sys.argv[1:]]))


if __name__ == "__main__":
    main()

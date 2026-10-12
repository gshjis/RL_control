"""Headless-графики — обёртка над ``python -m packages plot``."""

from __future__ import annotations

import sys

from packages.cli import main as cli_main


def main() -> None:
    """Строит и сохраняет график, передавая аргументы дальше в CLI."""
    raise SystemExit(cli_main(["plot", *sys.argv[1:]]))


if __name__ == "__main__":
    main()

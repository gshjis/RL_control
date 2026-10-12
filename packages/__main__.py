"""Позволяет запускать CLI как ``python -m packages <команда>``."""

from packages.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

"""Тесты CLI (packages/cli.py): разбор аргументов и переопределения пресета."""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.cli import (
    _discover_checkpoint,
    _experiment_from_args,
    build_parser,
    cmd_firmware,
    cmd_gui,
    cmd_plot,
)


def test_train_parses_overrides() -> None:
    """Команда train принимает переопределения и дефолтный чекпоинт."""
    args = build_parser().parse_args(
        ["train", "--total-timesteps", "123", "--n-envs", "4"]
    )

    assert args.handler.__name__ == "cmd_train"
    assert args.total_timesteps == 123
    assert args.n_envs == 4
    assert args.checkpoint.endswith("pendl")


def test_gui_and_plot_share_model_argument() -> None:
    """gui и plot принимают --model одной и той же семантики."""
    gui_args = build_parser().parse_args(["gui", "--model", "checkpoints/x"])
    plot_args = build_parser().parse_args(["plot", "--model", "checkpoints/x"])

    assert gui_args.handler is cmd_gui
    assert plot_args.handler is cmd_plot
    assert gui_args.model == plot_args.model == "checkpoints/x"


def test_rollout_commands_default_to_100_runs_20_seconds() -> None:
    """eval/plot без флагов используют 100 прогонов по 20 секунд."""
    args = build_parser().parse_args(["eval"])

    assert args.runs == 100
    assert args.seconds == 20.0
    assert args.output is None


def test_profile_is_headless_by_default() -> None:
    """profile не открывает окна, пока не передан --plot."""
    default_args = build_parser().parse_args(["profile"])
    plot_args = build_parser().parse_args(["profile", "--plot"])

    assert default_args.plot is False
    assert plot_args.plot is True


def test_experiment_from_args_applies_overrides() -> None:
    """Переопределения попадают в ExperimentConfig, остальное — из пресета."""
    args = build_parser().parse_args(
        ["train", "--total-timesteps", "7", "--n-envs", "3"]
    )

    experiment = _experiment_from_args(args)

    assert experiment.ppo.total_timesteps == 7
    assert experiment.n_envs == 3
    assert experiment.seed == 42


def test_firmware_requires_model_dir() -> None:
    """firmware без --model-dir — ошибка использования, а не дефолт."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["firmware"])


def test_firmware_defaults() -> None:
    """firmware по умолчанию собирает штатный скетч и печатает .hex в artifacts."""
    args = build_parser().parse_args(["firmware", "--model-dir", "/tmp/model"])

    assert args.handler is cmd_firmware
    assert args.model_dir == Path("/tmp/model")
    assert args.sketch.name == "pendulum_control"
    assert args.output_dir is None


def test_discover_checkpoint_returns_name_without_suffix(tmp_path: Path) -> None:
    """Итоговый чекпоинт <name>_model.zip превращается в префикс <name>."""
    (tmp_path / "pendl_model.zip").touch()
    (tmp_path / "pendl_config.json").touch()
    (tmp_path / "pendl_vecnormalize.pkl").touch()

    assert _discover_checkpoint(tmp_path) == tmp_path / "pendl"


def test_discover_checkpoint_accepts_validation_snapshot(tmp_path: Path) -> None:
    """Снимок валидации <name>.zip тоже поддерживается."""
    (tmp_path / "model_100000.zip").touch()

    assert _discover_checkpoint(tmp_path) == tmp_path / "model_100000"


def test_discover_checkpoint_rejects_ambiguous_folder(tmp_path: Path) -> None:
    """Два чекпоинта в папке — явная ошибка, а не случайный выбор."""
    (tmp_path / "a_model.zip").touch()
    (tmp_path / "b_model.zip").touch()

    with pytest.raises(SystemExit):
        _discover_checkpoint(tmp_path)


def test_discover_checkpoint_requires_existing_dir(tmp_path: Path) -> None:
    """Несуществующая папка — SystemExit с понятным сообщением."""
    with pytest.raises(SystemExit):
        _discover_checkpoint(tmp_path / "missing")


def test_unknown_command_is_a_usage_error() -> None:
    """Неизвестная команда — SystemExit с кодом 2, а не тихий пропуск."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nope"])

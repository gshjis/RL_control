"""Тесты CLI (packages/cli.py): разбор аргументов и переопределения пресета."""

from __future__ import annotations

import pytest

from packages.cli import _experiment_from_args, build_parser, cmd_gui, cmd_plot


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


def test_unknown_command_is_a_usage_error() -> None:
    """Неизвестная команда — SystemExit с кодом 2, а не тихий пропуск."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nope"])

"""Единая командная строка проекта: ``python -m packages <команда>``.

    python -m packages train  [--checkpoint PATH] [--total-timesteps N]
    python -m packages gui    [--model NAME]
    python -m packages plot   [--model NAME] [--runs N] [--seconds S] [--output PATH]
    python -m packages eval   [--model NAME] [--runs N] [--seconds S] [--output DIR]
    python -m packages profile [--duration S] [--plot]

Тяжёлые зависимости (Stable-Baselines3, Pygame, matplotlib) подгружаются
только внутри нужной команды, поэтому ``--help`` работает мгновенно.
Корневые скрипты (``PPO_train.py``, ``run_gui.py``, ``plot_creator.py``)
остались тонкими обёртками над этими командами.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from packages.experiment import ExperimentConfig
from packages.presets import default

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
PLOTS_ROOT = PROJECT_ROOT / "plots"

DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "checkpoints" / "ppo" / "pendl")
DEFAULT_MODEL = str(
    PROJECT_ROOT / "checkpoints" / "ppo" / "validation" / "model_1900000"
)


# ── Вспомогательные функции ───────────────────────────────────────────────
def _experiment_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Возвращает пресет с переопределениями из командной строки."""
    experiment = default()

    total_timesteps = getattr(args, "total_timesteps", None)
    if total_timesteps is not None:
        experiment = dataclasses.replace(
            experiment,
            ppo=dataclasses.replace(experiment.ppo, total_timesteps=total_timesteps),
        )

    n_envs = getattr(args, "n_envs", None)
    if n_envs is not None:
        experiment = dataclasses.replace(experiment, n_envs=n_envs)

    return experiment


def _run_rollout(args: argparse.Namespace):
    """Прогоняет политику по пресету и возвращает результат прогонов."""
    from packages.controllers.ppo.checkpoints import load_controller
    from packages.experiment import build_env
    from packages.rollout import rollout

    experiment = default()
    env = build_env(experiment, max_episode_steps=experiment.ppo.max_episode_steps)
    controller = load_controller(args.model)
    print(f"Загружена модель: {args.model}")
    return rollout(env, controller, seconds=args.seconds, runs=args.runs)


# ── Команды ───────────────────────────────────────────────────────────────
def cmd_train(args: argparse.Namespace) -> int:
    """Обучает PPO-агента по пресету и сохраняет чекпоинт."""
    from packages.controllers.ppo import PPOTrainer
    from packages.experiment import make_env_factory

    experiment = _experiment_from_args(args)

    print("Обучение PPO...")
    print(f"  total_timesteps = {experiment.ppo.total_timesteps}")
    print(f"  n_envs = {experiment.n_envs}")
    print(f"  Инерция двигателя: τ = {experiment.plant.motor_time_constant} с")

    trainer = PPOTrainer(
        ppo_config=experiment.ppo,
        controller_config=experiment.controller,
        env_factory=make_env_factory(experiment),
        n_envs=experiment.n_envs,
        checkpoint_dir=str(Path(args.checkpoint).parent / "validation"),
    )
    controller = trainer.fit()
    controller.save(args.checkpoint)

    print(f"Модель сохранена: {args.checkpoint}")
    print("Обучение PPO завершено!")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    """Запускает окно симуляции; без модели — ручное управление стрелками."""
    from packages.controllers.ppo.checkpoints import load_controller
    from packages.experiment import build_env
    from packages.simulation.gui import PendulumViewer

    experiment = default()
    env = build_env(experiment)

    try:
        controller = load_controller(args.model)
        print(f"Загружена модель: {args.model}")
    except FileNotFoundError:
        print(f"Модель {args.model} не найдена — ручное управление (←/→).")
        controller = None

    PendulumViewer(env=env, controller=controller).use()
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Сохраняет график среднего и разброса cos(theta1) по прогонам."""
    from packages.rollout import plot_rollout

    result = _run_rollout(args)

    output = args.output or (PLOTS_ROOT / "cos_theta1_mean_and_variance.png")
    path = plot_rollout(result, output)
    print(f"График сохранён: {path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Прогоняет политику, печатает метрики и сохраняет их вместе с графиком."""
    from packages.rollout import plot_rollout

    result = _run_rollout(args)
    summary = result.summary()

    print("Метрики прогона:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    out_dir = args.output or (ARTIFACTS_ROOT / "eval" / Path(args.model).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    plot_rollout(result, out_dir / "cos_plot.png")

    print(f"Результаты сохранены: {out_dir}")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Профилирует физическое ядро: скорость и погрешность датчика."""
    from profiling.profiling import profiler

    profiler(duration=args.duration, plot=args.plot)
    return 0


# ── Разбор аргументов ─────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """Возвращает готовый парсер CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m packages",
        description="Управление перевёрнутым маятником: обучение, GUI, оценка.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="обучить PPO-агента")
    train.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"путь для сохранения модели (по умолчанию: {DEFAULT_CHECKPOINT})",
    )
    train.add_argument(
        "--total-timesteps",
        type=int,
        dest="total_timesteps",
        default=None,
        help="переопределить число шагов обучения из пресета",
    )
    train.add_argument(
        "--n-envs",
        type=int,
        dest="n_envs",
        default=None,
        help="переопределить число параллельных сред",
    )
    train.set_defaults(handler=cmd_train)

    gui = subparsers.add_parser("gui", help="запустить визуализацию (Pygame)")
    gui.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="имя чекпоинта; если файла нет — ручное управление",
    )
    gui.set_defaults(handler=cmd_gui)

    rollout_parser = argparse.ArgumentParser(add_help=False)
    rollout_parser.add_argument("--model", default=DEFAULT_MODEL)
    rollout_parser.add_argument("--runs", type=int, default=100)
    rollout_parser.add_argument("--seconds", type=float, default=20.0)

    plot = subparsers.add_parser(
        "plot", parents=[rollout_parser], help="сохранить график cos(theta1)"
    )
    plot.add_argument("--output", type=Path, default=None, help="куда сохранить PNG")
    plot.set_defaults(handler=cmd_plot)

    evaluate = subparsers.add_parser(
        "eval", parents=[rollout_parser], help="оценить политику, метрики + график"
    )
    evaluate.add_argument(
        "--output", type=Path, default=None, help="каталог для metrics.json и PNG"
    )
    evaluate.set_defaults(handler=cmd_eval)

    profile = subparsers.add_parser(
        "profile", help="профилирование физического ядра"
    )
    profile.add_argument("--duration", type=float, default=2.0)
    profile.add_argument(
        "--plot",
        action="store_true",
        help="показать графики (открывает блокирующее окно)",
    )
    profile.set_defaults(handler=cmd_profile)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Разбирает аргументы и выполняет команду."""
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

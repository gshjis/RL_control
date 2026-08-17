"""Свободное падение маятника из верхнего положения.

Скрипт собирает два вида данных:

* real state — истинные координаты и скорости из физического движка;
* telemetry — выход блока датчиков после квантования, шума,
  дифференцирования и фильтрации.

Блок датчиков возвращает вектор из 8 элементов:
``[x, cos(theta1), sin(theta1), cos(theta2), sin(theta2),
dx/dt, dtheta1/dt, dtheta2/dt]``.
Углы в телеметрии не возвращаются напрямую — вместо них возвращаются
синус и косинус. Скорости в последних трёх элементах являются оценками
датчика, а не истинными значениями из физической модели.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from packages.simulation.CO import ObjectOfControl, PlantConfig, SensorConfig


def simulate_fall(
    *,
    dt: float = 5e-4,
    perturbation: float = 1e-3,
    bottom_tolerance: float = 2e-2,
    max_time: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Симулировать падение первого звена из положения ``theta1 = pi``.

    Нулевая сила приложена к тележке. Небольшое начальное отклонение нужно,
    потому что идеально перевёрнутый маятник с нулевой скоростью является
    математическим равновесием и самопроизвольно не начнёт падать.
    """

    plant_config = PlantConfig(
        M=1.0,
        m1=0.3,
        m2=0.0,
        l1=1.0,
        l2=0.0,
        g=-9.81,
        b_c=0.0,
        b_1=0.0,
        b_2=0.0,
        single_pendulum_mode=True,
        init_q_stats=np.array(
            [[0.0, np.pi + perturbation, 0.0], [0.0, 0.0, 0.0]],
            dtype=float,
        ),
        init_dq_stats=np.zeros((2, 3), dtype=float),
        dt=dt,
        motor_time_constant=0.0,
        mean_f=0.0,
        std_f=0.0,
    )

    # Шум оставлен включённым, чтобы на графике была видна разница между
    # истинным состоянием и измерением датчика. Для воспроизводимости seed.
    sensor_config = SensorConfig(
        encoder_resolution_1=4096,
        encoder_resolution_2=4096,
        cart_sensor_resolution=1e-4,
        noise_std_q=(5e-4, 2e-3, 2e-3),
        noise_std_dq=(0.0, 0.0, 0.0),
        seed=42,
        noise_pool_size=100_000,
        differentiator_cutoff_hz=None,
        filter_cutoff_hz=50.0,
    )
    pendulum = ObjectOfControl(plant_config, sensor_config)

    times: list[float] = []
    real_states: list[np.ndarray] = []
    telemetry: list[np.ndarray] = []

    bottom_reached = False
    max_steps = int(max_time / dt)
    for step in range(max_steps):
        # Сила равна нулю: маятник падает под действием гравитации.
        pendulum.update_physics(F_ideal=0.0, n_updates=1)

        real_state = pendulum.get_clean_state()
        measured = np.asarray(pendulum.get_telemetry(), dtype=float)

        times.append((step + 1) * dt)
        real_states.append(real_state)
        telemetry.append(measured)

        # В C++ углы нормализуются в [0, 2*pi). Нижнее положение — theta=0
        # (эквивалентно theta=2*pi), поэтому используем циклическую ошибку.
        theta = real_state[1]
        angle_to_bottom = abs(np.arctan2(np.sin(theta), np.cos(theta)))
        if angle_to_bottom <= bottom_tolerance:
            bottom_reached = True
            break

    if not bottom_reached:
        raise RuntimeError(
            f"Маятник не достиг нижнего положения за {max_time:.2f} с; "
            "увеличьте max_time или проверьте параметры модели."
        )

    return (
        np.asarray(times),
        np.asarray(real_states),
        np.asarray(telemetry),
    )


def plot_results(
    times: np.ndarray,
    real_states: np.ndarray,
    measured: np.ndarray,
) -> None:
    """Построить истинные координаты/скорости и все каналы телеметрии."""

    # Нормализация угла в движке создаёт скачок 2*pi -> 0; unwrap делает
    # график непрерывным и не изменяет сами собранные реальные данные.
    real_theta1 = np.unwrap(real_states[:, 1])
    real_theta2 = np.unwrap(real_states[:, 2])

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(times, real_states[:, 0], label="real x")
    axes[0].plot(times, real_theta1, label="real theta1")
    axes[0].plot(times, real_theta2, label="real theta2")
    axes[0].set_ylabel("q, м / рад")
    axes[0].set_title("Истинные обобщённые координаты")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(times, real_states[:, 3], label="real dx/dt")
    axes[1].plot(times, real_states[:, 4], label="real dtheta1/dt")
    axes[1].plot(times, real_states[:, 5], label="real dtheta2/dt")
    axes[1].set_ylabel("dq, м/с / рад/с")
    axes[1].set_title("Истинные обобщённые скорости")
    axes[1].grid(True)
    axes[1].legend()

    telemetry_labels = (
        "telemetry x",
        "telemetry cos(theta1)",
        "telemetry sin(theta1)",
        "telemetry cos(theta2)",
        "telemetry sin(theta2)",
        "telemetry dx/dt",
        "telemetry dtheta1/dt",
        "telemetry dtheta2/dt",
    )
    for index, label in enumerate(telemetry_labels):
        axes[2].plot(times, measured[:, index], label=label)
    axes[2].set_xlabel("Время, с")
    axes[2].set_ylabel("измеренное значение")
    axes[2].set_title("Телеметрия SensorBlock (8 каналов)")
    axes[2].grid(True)
    axes[2].legend(ncol=2, fontsize="small")

    fig.tight_layout()
    print(f"Симуляция завершена в t = {times[-1]:.4f} с")
    print(f"Размер real state: {real_states.shape}; размер telemetry: {measured.shape}")
    print("Формат telemetry: [x, cos(theta1), sin(theta1), cos(theta2), "
          "sin(theta2), dx/dt, dtheta1/dt, dtheta2/dt]")
    plt.show()
    velocity_error = np.cos(real_states[:, 1]) - measured[:, 1]
    print("velocity L2:", np.linalg.norm(velocity_error))
    print("velocity RMSE:", np.sqrt(np.mean(velocity_error**2)))



def main() -> None:
    times, real_states, measured = simulate_fall()
    plot_results(times, real_states, measured)


if __name__ == "__main__":
    main()

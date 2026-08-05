"""Визуализация и управление симуляцией перевёрнутого маятника (pygame)."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

import numpy as np
import pygame

from packages.simulation.ENV.env import PendulumEnv
from packages.simulation.GUI import constants as C


class PendulumViewer:
    """
    Окно симуляции перевёрнутого маятника.

    Принимает одну среду (``PendulumEnv``) и опциональный контроллер.
    Если контроллер не задан — используется ручное управление стрелками.

    Главный цикл ``use()``:
    - отображает окно и рисует маятник согласно конфигурации (удлинение
      стержня и масса отображаются как удлинение и размер точки на конце);
    - работает на ``FPS = 120``;
    - сбрасывает среду (возвращает показания датчиков);
    - вычисляет управление контроллером и передаёт его в ``step()``;
    - использует аккумулятор времени, чтобы симуляционное время совпадало
      с реальным (компьютерным) временем.
    """

    def __init__(
        self,
        env: PendulumEnv,
        controller: Any,
    ) -> None:
        self._env = env
        self._controller = controller

        # Параметры растения для отрисовки.
        self._plant = env._plant
        self._l1: float = float(self._plant._l1)
        self._l2: float = float(self._plant._l2)
        self._m1: float = float(self._plant._m1)
        self._m2: float = float(self._plant._m2)

        # Шаг симуляции (такт контроллера).
        self._sim_dt: float = float(getattr(env, "_controller_dt", 0.01))

        pygame.init()
        self._screen = pygame.display.set_mode((C.WIDTH, C.HEIGHT))
        pygame.display.set_caption("Inverted Pendulum")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 16)
        self._small_font = pygame.font.SysFont("monospace", 13)

        self._running = True
        self._manual_force = 0.0
        self._time_acc = 0.0
        self._sim_time = 0.0
        self._last_action = 0.0
        self._obs: np.ndarray | None = None
        self._reward = 0.0

        # Буферы для графиков.
        self._sine1_hist: deque[float] = deque(maxlen=200)
        self._sine2_hist: deque[float] = deque(maxlen=200)
        self._err_hist: deque[float] = deque(maxlen=200)

    # ── Главный цикл ──────────────────────────────────────────────────────
    def use(self) -> None:
        """Запустить окно симуляции (блокирующий цикл)."""
        self._reset()
        last = time.perf_counter()

        while self._running:
            now = time.perf_counter()
            dt_real = now - last
            last = now

            self._handle_events()

            # Аккумулятор времени: симуляция идёт в реальном времени.
            self._time_acc += dt_real
            while self._time_acc >= self._sim_dt:
                self._step()
                self._time_acc -= self._sim_dt

            self._draw()
            pygame.display.flip()
            self._clock.tick(C.FPS)

        pygame.quit()

    # ── Сброс ─────────────────────────────────────────────────────────────
    def _reset(self) -> None:
        """Сбросить симуляцию.

        Если контроллера нет — маятник стартует отклонённым от нижнего
        положения и колеблется свободно (сила = 0).
        """
        if self._controller is None:
            self._obs = self._plant.get_telemetry()
        else:
            self._obs, _ = self._env.reset()
        self._sim_time = 0.0
        self._sine1_hist.clear()
        self._sine2_hist.clear()
        self._err_hist.clear()

    # ── Шаг симуляции ─────────────────────────────────────────────────────
    def _step(self) -> None:
        if self._controller is not None:
            action = self._controller.action(self._obs)
        else:
            action = np.array([self._manual_force], dtype=np.float64)
        self._last_action = float(np.asarray(action).reshape(-1)[0])
        self._obs, self._reward, terminated, truncated, _ = self._env.step(action)

        self._sim_time += self._sim_dt

        # Логирование графиков.
        q = self._plant.q
        self._sine1_hist.append(float(np.sin(q[1])))
        self._sine2_hist.append(float(np.sin(q[2])))
        self._err_hist.append(float(q[0] - self._env._target(self._time_acc)[0]))
        if terminated:
            print(terminated)
            self._reset()

    # ── Обработка ввода ───────────────────────────────────────────────────
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._running = False
                elif event.key == pygame.K_SPACE:
                    self._reset()

        keys = pygame.key.get_pressed()
        self._manual_force = 0.0
        if keys[pygame.K_LEFT]:
            self._manual_force = -C.FORCE_PER_FRAME
        if keys[pygame.K_RIGHT]:
            self._manual_force = C.FORCE_PER_FRAME

    # ── Отрисовка ─────────────────────────────────────────────────────────
    def _draw(self) -> None:
        self._screen.fill(C.BLACK)
        self._draw_track()
        self._draw_pendulum()
        self._draw_force_arrow()
        self._draw_graphs()
        self._draw_hud()

    def _draw_track(self) -> None:
        pygame.draw.line(
            self._screen, C.GRAY, (0, C.TRACK_Y), (C.WIDTH, C.TRACK_Y), 2
        )

    def _draw_pendulum(self) -> None:
        q = self._plant.q
        x = float(q[0])
        th1 = float(q[1])
        th2 = float(q[2])

        pivot_x = C.WIDTH / 2.0 + x * C.SCALE
        pivot_y = float(C.TRACK_Y)

        # Тележка.
        cart_rect = pygame.Rect(
            0, 0, C.CART_W, C.CART_H
        )
        cart_rect.center = (int(pivot_x), int(pivot_y))
        pygame.draw.rect(self._screen, C.WHITE, cart_rect, 2)

        # Колёса.
        for dx in (-C.CART_W / 3.0, C.CART_W / 3.0):
            pygame.draw.circle(
                self._screen,
                C.GRAY,
                (int(pivot_x + dx), int(pivot_y + C.CART_H / 2.0)),
                C.WHEEL_R,
            )

        # Первый стержень.
        end1_x = pivot_x + self._l1 * np.sin(th1) * C.SCALE
        end1_y = pivot_y + self._l1 * np.cos(th1) * C.SCALE
        pygame.draw.line(
            self._screen,
            C.ORANGE,
            (int(pivot_x), int(pivot_y)),
            (int(end1_x), int(end1_y)),
            4,
        )

        # Масса первого стержня (размер пропорционален массе).
        r1 = max(3.0, C.PEND_R * (1.0 + self._m1))
        pygame.draw.circle(
            self._screen, C.RED, (int(end1_x), int(end1_y)), int(r1)
        )

        # Второй стержень (если есть).
        if self._l2 > 0.0:
            end2_x = end1_x + self._l2 * np.sin(th1 + th2) * C.SCALE
            end2_y = end1_y + self._l2 * np.cos(th1 + th2) * C.SCALE
            pygame.draw.line(
                self._screen,
                C.GREEN,
                (int(end1_x), int(end1_y)),
                (int(end2_x), int(end2_y)),
                3,
            )
            r2 = max(3.0, C.PEND_R * (1.0 + self._m2))
            pygame.draw.circle(
                self._screen, C.GREEN, (int(end2_x), int(end2_y)), int(r2)
            )

    def _draw_force_arrow(self) -> None:
        pivot_x = C.WIDTH / 2.0 + float(self._plant.q[0]) * C.SCALE
        length = self._last_action * C.FORCE_SCALE
        start = (int(pivot_x), int(C.TRACK_Y - C.CART_H))
        end = (int(pivot_x + length), int(C.TRACK_Y - C.CART_H))
        pygame.draw.line(self._screen, C.GREEN, start, end, 3)
        if abs(length) > 1.0:
            direction = 1.0 if length > 0 else -1.0
            tip = (end[0] + int(direction * 8), end[1])
            pygame.draw.circle(self._screen, C.GREEN, tip, 4)

    def _draw_graphs(self) -> None:
        # График sin(θ₁) и sin(θ₂).
        self._draw_plot(
            C.SINE_GRAPH_X,
            C.SINE_GRAPH_Y,
            C.SINE_GRAPH_W,
            C.SINE_GRAPH_H,
            C.SINE_BG,
            C.SINE_GRID,
            [(self._sine1_hist, C.SINE_COLOR), (self._sine2_hist, C.SINE_COLOR2)],
            label="sin(theta)",
        )
        # График ошибки по X.
        self._draw_plot(
            C.ERR_GRAPH_X,
            C.ERR_GRAPH_Y,
            C.ERR_GRAPH_W,
            C.ERR_GRAPH_H,
            C.ERR_BG,
            C.ERR_GRID,
            [(self._err_hist, C.ERR_COLOR)],
            label="err_x",
        )

    def _draw_plot(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        bg: tuple[int, int, int],
        grid: tuple[int, int, int],
        series: list[tuple[deque[float], tuple[int, int, int]]],
        label: str,
    ) -> None:
        pygame.draw.rect(self._screen, bg, (x, y, w, h))
        pygame.draw.rect(self._screen, grid, (x, y, w, h), 1)
        # Средняя линия.
        mid = y + h // 2
        pygame.draw.line(self._screen, grid, (x, mid), (x + w, mid), 1)

        for data, color in series:
            if len(data) < 2:
                continue
            n = len(data)
            step = w / (n - 1)
            amp = h / 2.0 - 4
            pts = []
            for i, v in enumerate(data):
                px = x + i * step
                py = mid - float(np.clip(v, -1.0, 1.0)) * amp
                pts.append((int(px), int(py)))
            pygame.draw.lines(self._screen, color, False, pts, 2)

        self._screen.blit(
            self._small_font.render(label, True, C.WHITE), (x + 4, y + 4)
        )

    def _draw_hud(self) -> None:
        q = self._plant.q
        dq = self._plant.dq
        lines = [
            f"t = {self._sim_time:6.2f} s",
            f"x  = {q[0]:+7.3f} m",
            f"th1= {q[1]:+7.3f} rad",
            f"th2= {q[2]:+7.3f} rad",
            f"dx = {dq[0]:+7.3f} m/s",
            f"F  = {self._last_action:+7.2f} N",
            f"reward = {self._reward:+.3f}",
        ]
        if self._controller is None:
            lines.append("MANUAL (arrows)")
        else:
            lines.append(f"CTRL: {self._controller.name}")

        for i, line in enumerate(lines):
            surf = self._font.render(line, True, C.WHITE)
            self._screen.blit(surf, (15, 15 + i * 20))

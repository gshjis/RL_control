"""
Pygame-визуализация перевёрнутого маятника.
"""

from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np
import pygame

from packages.simulation.ENV.env import PendulumEnv

from .constants import *
from .dialogs import ask_recording, ask_save_video
from .draw import (
    draw_cart,
    draw_controller_button,
    draw_error_graph,
    draw_force_arrow,
    draw_hud,
    draw_pendulums,
    draw_sine_graph,
    draw_target_marker,
)
from .event_controller import EventController
from .recorder import compile_video

# ═══════════════════════════════════════════════════════════════════════════
# PendulumViewer
# ═══════════════════════════════════════════════════════════════════════════

class PendulumViewer:
    """
    Pygame-визуализация перевёрнутого маятника на тележке.

    Использует ``PendulumEnv.step()`` для корректного тактирования
    управления с имитацией вычислительной задержки.
    Отрисовка — 60 FPS. Симуляция идёт в реальном времени:
    количество вызовов ``env.step()`` за кадр определяется накопленным
    временем относительно ``controller._dt``.
    """

    def __init__(
        self,
        env: PendulumEnv,
    ) -> None:
        self._env = env
        self._F: float = 0.0  # Текущая сила, применяемая к маятнику

        # Pygame
        pygame.init()
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Consolas", 16, bold=True)

        title = "Перевёрнутый маятник"
        title += f" — {self._env.controller.name}" if self._env.controller else " — ручное управление"
        title += "  (Пробел сброс, Q / ESC выход)"
        pygame.display.set_caption(title)

        self._recording: bool = False
        self._record_dir: str = os.path.abspath(".")
        self._frame_index: int = 0
        self._need_compile: bool = False

        self._event_controller = EventController(self._screen, self._clock)

        # Target marker
        from .constants import (
            MARKER_COLOR,
            MARKER_H,
            MARKER_HOVER_COLOR,
            MARKER_MAX_X,
            MARKER_MIN_X,
            MARKER_SPEED,
            MARKER_THROTTLE_MS,
            MARKER_W,
        )

        self._marker_color = MARKER_COLOR
        self._marker_hover_color = MARKER_HOVER_COLOR
        self._marker_speed = MARKER_SPEED
        self._marker_throttle_ms = MARKER_THROTTLE_MS
        self._marker_min_x = MARKER_MIN_X
        self._marker_max_x = MARKER_MAX_X
        self._marker_w = MARKER_W
        self._marker_h = MARKER_H
        self._marker_x = float(self._env.target_state[0])
        self._marker_dragging = False
        self._drag_offset_x = 0
        self._last_marker_update_ms = 0
        self._controller_enabled = True if self._env.controller is not None else False
        self._controller = self._env.controller
        self._controller_backup = None

        # Аккумулятор симуляционного времени (сек).
        # На каждом кадре добавляем реально прошедшее время dt_sec,
        # и вызываем env.step(), пока накопление >= controller._dt.
        self._sim_accumulator: float = 0.0

        # Последнее observation (отфильтрованное) для get_control()
        self._obs: np.ndarray | None = None

        # Буферы для графиков (deque — для отрисовки, ограниченный размер)
        self._sin1_history: deque[float] = deque(maxlen=800)
        self._sin2_history: deque[float] = deque(maxlen=800)
        self._err_history: deque[float] = deque(maxlen=800)

        # Полная история для CSV-лога (неограниченная)
        self._csv_time: list[float] = []
        self._csv_sin1: list[float] = []
        self._csv_sin2: list[float] = []
        self._csv_err_x: list[float] = []

    # ── Публичный метод ───────────────────────────────────────────────────

    def use(self) -> None:
        """Запустить главный цикл визуализации (блокирующий)."""
        # Диалог записи
        if ask_recording(self._screen, self._clock):
            self._recording = True
            self._record_dir = os.path.abspath(".")
            self._frame_index = 0
            self._need_compile = False
        else:
            self._recording = False

        running = True
        manual_force = 0.0
        force_per_frame = FORCE_PER_FRAME

        self._start_ticks = pygame.time.get_ticks()
        prev_ticks = pygame.time.get_ticks()
        last_save_acc_ms = 0
        while running:
            now = pygame.time.get_ticks()
            dt_ms = now - prev_ticks
            prev_ticks = now
            dt_sec = dt_ms / 1000.0
            

            # ── 1. ОБРАБОТКА СОБЫТИЙ ──────────────────────────────────
            actions = self._event_controller.poll()
            

            # Стрелки для перемещения цели (только при включённом контроллере)
            if self._controller is not None:
                keys = pygame.key.get_pressed()
                changed = False
                if keys[pygame.K_LEFT]:
                    if self._marker_min_x is not None:
                        self._marker_x = max(self._marker_min_x, self._marker_x - 0.05)
                    else:
                        self._marker_x -= 0.05
                    changed = True
                if keys[pygame.K_RIGHT]:
                    if self._marker_max_x is not None:
                        self._marker_x = min(self._marker_max_x, self._marker_x + 0.05)
                    else:
                        self._marker_x += 0.05
                    changed = True
                if changed:
                    self._env.target_state[0] = float(self._marker_x)
                    self._last_marker_update_ms = now

            self._last_events = actions.get("events", [])

            if not actions.get("running", True):
                running = False
                if actions.get("quit", False):
                    break
            

            # Переключение записи
            if actions.get("toggle_record", False):
                self._toggle_recording()

            # Переключение контроллера (клавиша C)
            if actions.get("toggle_controller", False):
                self._toggle_controller()
            

            # Переключение контроллера по клику на кнопке
            mx_my = actions.get("mouse_pos")
            if mx_my:
                mx, my = mx_my
                ctrl_btn = (WIDTH - 150, 10, 120, 30)
                if (
                    ctrl_btn[0] <= mx <= ctrl_btn[0] + ctrl_btn[2]
                    and ctrl_btn[1] <= my <= ctrl_btn[1] + ctrl_btn[3]
                ):
                    self._toggle_controller()
            

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
                running = False

            # ── 2. СИМУЛЯЦИЯ ───────────────────────────────────────────
            if self._controller is not None:
                self._sim_accumulator += dt_sec
                dt_ctrl = self._controller.dt
                # Инициализация obs (первый шаг) — отфильтрованное
                if self._obs is None:
                    self._obs, _ = self._env.reset()
                while self._sim_accumulator >= dt_ctrl:
                    target_s = self._env.target_state
                    # get_control() получает отфильтрованное состояние (без двойной фильтрации)
                    action = self._controller.get_control(self._obs[:6], target_s)
                    self._obs, r, terminated, truncated, info = self._env.step(action)
                    self._sim_accumulator -= dt_ctrl
                    # Автоматический сброс при завершении эпизода
                    if terminated or truncated:
                        self._reset()
                        break
            else:
                # ── Ручное управление (контроллер отключён) ──────────
                self._sim_accumulator += dt_sec
                dt_ctrl = self._env.controller.dt if self._env.controller is not None else 0.01
                while self._sim_accumulator >= dt_ctrl:
                    # Клавиши: ←/→ — сила влево/вправо (инкрементально)
                    if keys[pygame.K_LEFT]:
                        manual_force -= force_per_frame
                    if keys[pygame.K_RIGHT]:
                        manual_force += force_per_frame
                    self._env.step(manual_force)
                    self._sim_accumulator -= dt_ctrl

            # ── 3. СБРОС ──────────────────────────────────────────────
            if keys[pygame.K_SPACE]:
                self._reset()
        
            # ── 4. ОТРИСОВКА ────────────────────────────────────────────
            self._draw(self._env.plant.motor_force)
            self._handle_marker_events()
            

            # ── 5. ЗАПИСЬ ВИДЕО ──────────────────────────────────────
            if self._recording:
                last_save_acc_ms += dt_ms
                last_save_acc_ms = self._save_frame_if_recording(last_save_acc_ms)
            

            # Сборка видео после остановки записи
            if not self._recording and self._need_compile and self._record_dir is not None:
                self._compile_video_dir()
                self._need_compile = False
            

            self._clock.tick(FPS)



        # ── Выход: сохранение видео ──────────────────────────────────
        if self._record_dir is not None:
            try:
                import glob
                frames = sorted(glob.glob(os.path.join(self._record_dir, "frame_*.png")))
            except Exception:
                frames = []

            if frames and ask_save_video(self._screen, self._clock, len(frames)):
                try:
                    sim_fps = self._compute_sim_fps()
                    if sim_fps > 0:
                        compile_video(self._record_dir, sim_fps)
                        for p in frames:
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                except Exception:
                    pass

        # Сохранение CSV-логов

        pygame.quit()
        sys.exit(0)

    # ── Вспомогательные методы ──────────────────────────────────────────

    def _toggle_controller(self) -> None:
        """Включить/выключить контроллер."""
        self._controller_enabled = not self._controller_enabled

        if not self._controller_enabled:
            self._controller_backup = self._env.controller
            if self._env.controller is not None and hasattr(self._env.controller, "reset"):
                try:
                    self._env.controller.reset()
                except Exception:
                    pass
            self._controller = None
        else:
            self._controller = self._controller_backup
            if self._controller is not None and hasattr(self._controller, "reset"):
                try:
                    self._controller.reset()
                except Exception:
                    pass
        # Сбросить аккумулятор и силу при переключении
        self._sim_accumulator = 0.0
        self._F = 0.0

    def _toggle_recording(self) -> None:
        """Включить/выключить запись кадров."""
        import tempfile

        if not self._recording:
            self._record_dir = tempfile.mkdtemp(prefix="pendulum_rec_", dir=os.path.abspath("."))
            self._frame_index = 0
            self._recording = True
            self._need_compile = False
        else:
            if self._record_dir is not None:
                fname = f"frame_{self._frame_index:06d}.png"
                path = os.path.join(self._record_dir, fname)
                try:
                    pygame.image.save(self._screen, path)
                    self._frame_index += 1
                except Exception:
                    pass
                if self._frame_index > 0:
                    self._need_compile = True
            self._recording = False

    def _save_frame_if_recording(self, last_save_acc_ms: int) -> int:
        """Сохранить кадр если накоплено достаточно времени."""
        # Интервал сохранения — 1/FPS, т.е. каждый кадр (60 FPS).
        frame_interval_ms = int(1000.0 / FPS)

        if (
            last_save_acc_ms >= frame_interval_ms
            and self._recording
            and self._record_dir is not None
        ):
            last_save_acc_ms = last_save_acc_ms % frame_interval_ms
            fname = f"frame_{self._frame_index:06d}.png"
            path = os.path.join(self._record_dir, fname)
            try:
                pygame.image.save(self._screen, path)
                self._frame_index += 1
            except Exception:
                pass
        return last_save_acc_ms

    def _compile_video_dir(self) -> None:
        """Собрать видео из кадров."""
        try:
            sim_fps = self._compute_sim_fps()
            if sim_fps > 0:
                compile_video(self._record_dir, sim_fps)
        except Exception:
            pass
        finally:
            self._need_compile = False

    def _compute_sim_fps(self) -> int:
        """Вычислить FPS симуляции для видео."""
        try:
            import glob

            frames = sorted(glob.glob(os.path.join(self._record_dir, "frame_*.png")))
            n_frames = len(frames)

            sim_seconds = (pygame.time.get_ticks() - self._start_ticks) / 1000.0

            if sim_seconds > 0 and n_frames > 0:
                return max(1, round(n_frames / sim_seconds))
        except Exception:
            pass

        # Fallback: 60 FPS
        return FPS

    def _save_csv(self) -> None:
        """Сохранить историю симуляции в CSV-файлы."""
        if not self._csv_time:
            return

    def _reset(self) -> None:
        """Сброс состояния симуляции."""
        self._env._plant.reset()
        if self._env.controller is not None:
            self._env.controller.reset()
        self._start_ticks = pygame.time.get_ticks()
        self._sim_accumulator = 0.0
        self._F = 0.0
        self._obs = None
        # Очистка буферов графиков
        self._sin1_history.clear()
        self._sin2_history.clear()
        self._err_history.clear()
        # CSV-логи не очищаем — накапливаем за всё время сессии

    def _handle_marker_events(self) -> None:
        """Обработка перетаскивания маркера цели."""
        events = getattr(self, "_last_events", [])
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                cart_x_px = int(WIDTH // 2 + self._env._plant.q[0] * SCALE)
                cart_y_px = TRACK_Y - CART_H // 2
                rect = pygame.Rect(
                    cart_x_px - self._marker_w // 2,
                    cart_y_px - self._marker_h,
                    self._marker_w,
                    self._marker_h,
                )
                if rect.collidepoint(mx, my) and self._controller is not None:
                    self._marker_dragging = True
                    self._drag_offset_x = mx - rect.x

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self._marker_dragging:
                    self._marker_dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self._marker_dragging:
                    mx, _ = event.pos
                    new_px = mx - self._drag_offset_x
                    new_x = (new_px - WIDTH // 2) / SCALE

                    if self._marker_min_x is not None:
                        new_x = max(self._marker_min_x, new_x)
                    if self._marker_max_x is not None:
                        new_x = min(self._marker_max_x, new_x)

                    now_ms = pygame.time.get_ticks()
                    dt = max(1, now_ms - self._last_marker_update_ms) / 1000.0
                    max_dx = self._marker_speed * dt
                    dx = new_x - self._marker_x
                    if abs(dx) > max_dx:
                        dx = max_dx if dx > 0 else -max_dx

                    self._marker_x += dx
                    self._last_marker_update_ms = now_ms
                    self._env.target_state[0] = float(self._marker_x)

    # ── Отрисовка ─────────────────────────────────────────────────────────

    def _draw(self, applied_force: float) -> None:
        """Отрисовка сцены."""
        q = self._env._plant.q
        dq = self._env._plant.dq
        is_single = self._env._plant.single_pendulum_mode

        x = q[0]
        th1 = q[1]
        th2 = q[2] if not is_single else 0.0
        dx = dq[0]
        dth1 = dq[1]
        dth2 = dq[2] if not is_single else 0.0

        cart_x_px = int(WIDTH // 2 + x * SCALE)
        cart_y_px = TRACK_Y - CART_H // 2

        self._screen.fill(BLACK)
        pygame.draw.line(self._screen, GRAY, (0, TRACK_Y), (WIDTH, TRACK_Y), 2)

        draw_cart(self._screen, cart_x_px, cart_y_px)
        draw_pendulums(
            self._screen, cart_x_px, cart_y_px,
            th1, th2, is_single,
            l1=self._env._plant._l1, l2=self._env._plant._l2,
        )
        draw_force_arrow(self._screen, applied_force, cart_x_px, cart_y_px)

        # HUD
        mode = "PID" if self._env.controller is not None else "РУЧНОЕ"
        gains_str = ""
        if self._env.controller is not None and hasattr(self._env.controller, "gains"):
            g = self._env.controller.gains
            gains_str = f"  Kp={g[0]:.1f}  Ki={g[1]:.1f}  Kd={g[2]:.1f}  Kx={g[3]:.1f}"

        lines = [
            f"Сила: {applied_force:+.1f} Н  [{mode}]",
            f"x  = {x:+.3f} м      θ1 = {np.degrees(th1):+7.1f}°",
        ]
        if not is_single:
            lines.append(f"θ2 = {np.degrees(th2):+7.1f}°")
        lines.append(f"ẋ  = {dx:+.3f} м/с   θ̇1 = {np.degrees(dth1):+7.1f}°/с")
        if not is_single:
            lines.append(f"θ̇2 = {np.degrees(dth2):+7.1f}°/с")
        if gains_str:
            lines.append(gains_str)

        draw_hud(self._screen, self._font, lines)
        draw_controller_button(self._screen, self._font, self._controller_enabled)

        # Маркер цели
        if self._env.controller is not None:
            marker_px = int(WIDTH // 2 + self._marker_x * SCALE)
            marker_py = cart_y_px
            mx, my = pygame.mouse.get_pos()
            rect = pygame.Rect(
                marker_px - self._marker_w // 2,
                marker_py - self._marker_h,
                self._marker_w,
                self._marker_h,
            )
            color = self._marker_hover_color if rect.collidepoint(mx, my) else self._marker_color
            draw_target_marker(
                self._screen,
                marker_px,
                marker_py,
                color,
                self._marker_w,
                self._marker_h,
                f"{self._marker_x:.3f}",
            )

        # Время
        elapsed_s = self._get_elapsed_time()
        time_surf = self._font.render(f"Время: {elapsed_s:.2f} с", True, GREEN)
        self._screen.blit(time_surf, (WIDTH - 360, 20))

        # ── Графики ─────────────────────────────────────────────────────
        # Текущее время симуляции
        sim_time = self._get_elapsed_time()

        # sin(θ)
        sin_th1 = np.sin(th1)
        sin_th2 = np.sin(th2) if not is_single else 0.0
        self._sin1_history.append(sin_th1)
        if not is_single:
            self._sin2_history.append(sin_th2)

        # Ошибка по X
        err_x = self._env.target_state[0] - x
        self._err_history.append(err_x)

        # Логирование в CSV (полная история)
        self._csv_time.append(sim_time)
        self._csv_sin1.append(sin_th1)
        self._csv_sin2.append(sin_th2)
        self._csv_err_x.append(err_x)

        # Отрисовка графика sin(θ)
        draw_sine_graph(
            self._screen, self._font,
            list(self._sin1_history),
            list(self._sin2_history) if not is_single else None,
            is_single,
        )

        # Отрисовка графика ошибки
        draw_error_graph(
            self._screen, self._font,
            list(self._err_history),
        )

        pygame.display.flip()

    def _get_elapsed_time(self) -> float:
        """Возвращает прошедшее время симуляции в секундах."""
        if hasattr(self, "_start_ticks"):
            return (pygame.time.get_ticks() - self._start_ticks) / 1000.0
        return 0.0
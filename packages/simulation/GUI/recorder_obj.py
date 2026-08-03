"""Recorder: объект-обёртка для логики записи и сборки видео."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from .recorder import compile_video



class Recorder:
    def __init__(self, record_dir: Optional[str] = None) -> None:
        self.record_dir = record_dir or os.path.abspath(".")
        self.frame_index = 0
        self.recording = False
        self.need_compile = False

    def start(self) -> None:
        d = tempfile.mkdtemp(prefix="pendulum_rec_", dir=self.record_dir)
        self.record_dir = d
        self.frame_index = 0
        self.recording = True
        self.need_compile = True

    def stop(self) -> None:
        self.recording = False

    def save_frame(self, screen) -> None:
        if not self.recording or self.record_dir is None:
            return
        fname = f"frame_{self.frame_index:06d}.png"
        path = os.path.join(self.record_dir, fname)
        try:
            import pygame

            pygame.image.save(screen, path)
            self.frame_index += 1
        except Exception:
            pass

    def compile(self, fps: int) -> Optional[str]:
        if self.record_dir is None:
            return None
        out = compile_video(self.record_dir, fps)
        return out


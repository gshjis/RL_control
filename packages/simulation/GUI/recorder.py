"""Логика записи кадров и сборки видео для PendulumViewer."""

from __future__ import annotations

import os
import subprocess
from typing import Optional



def compile_video(record_dir: str, fps: int) -> Optional[str]:
    """Собрать видео из PNG-фреймов с помощью ffmpeg. Возвращает путь к mp4 или None."""
    try:
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i",
            os.path.join(record_dir, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "output.mp4",
        ]
        subprocess.run(cmd, check=True, cwd=record_dir)
        out_mp4 = os.path.join(record_dir, "output.mp4")
        return out_mp4
    except Exception:
        return None


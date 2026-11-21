"""
fps_counter.py
--------------
A lightweight, rolling-window FPS counter. Uses a fixed-size deque of
timestamps rather than a single-frame delta so the reported FPS is stable
and not noisy from one slow/fast frame.
"""

import time
from collections import deque


class FPSCounter:
    """Rolling-window frames-per-second counter."""

    def __init__(self, window_size: int = 30):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.window_size = window_size
        self._timestamps = deque(maxlen=window_size)

    def tick(self) -> None:
        """Call once per processed frame."""
        self._timestamps.append(time.perf_counter())

    def get_fps(self) -> float:
        """Return the current smoothed FPS estimate (0.0 if not enough data yet)."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()

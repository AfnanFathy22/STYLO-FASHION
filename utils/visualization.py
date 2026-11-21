"""
visualization.py
-----------------
Drawing/compositing helpers shared by main.py: FPS readout, panel labels,
a 2x2 combined display grid (original+pose | segmentation | mesh | composite),
and a brief "saved!" flash overlay.
"""

from typing import Dict

import cv2
import numpy as np

import config


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw an FPS counter with a small background box for readability."""
    text = f"FPS: {fps:.1f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = 10, 10
    cv2.rectangle(frame, (x - 4, y - 4), (x + tw + 4, y + th + baseline + 4), config.COLOR_FPS_BG, -1)
    cv2.putText(frame, text, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2, cv2.LINE_AA)
    return frame


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    """Add a labeled bar at the top of a panel (e.g. 'Pose', 'Segmentation')."""
    h, w = frame.shape[:2]
    bar = np.zeros((config.PANEL_LABEL_HEIGHT, w, 3), dtype=np.uint8)
    cv2.putText(
        bar, label, (8, config.PANEL_LABEL_HEIGHT - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_TEXT, 1, cv2.LINE_AA,
    )
    return np.vstack([bar, frame])


def draw_save_flash(frame: np.ndarray, message: str = "SAVED") -> np.ndarray:
    """Draw a short-lived confirmation banner (used for a few frames after 'S' is pressed)."""
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    x = (w - tw) // 2
    y = 60
    cv2.rectangle(frame, (x - 15, y - th - 15), (x + tw + 15, y + 15), config.COLOR_SAVE_FLASH, -1)
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    return frame


def make_panel_grid(panels: Dict[str, np.ndarray], scale: float = None) -> np.ndarray:
    """
    Arrange named panels (e.g. {'Pose': img1, 'Segmentation': img2, ...}) into
    a 2x2 (or Nx2) labeled grid for a single combined display window.

    All panels are resized to the same dimensions before stacking, so panels
    of slightly different sizes (e.g. due to different processing paths)
    won't break the layout.
    """
    scale = config.DISPLAY_SCALE if scale is None else scale
    names = list(panels.keys())
    if not names:
        raise ValueError("No panels provided to make_panel_grid")

    ref_h, ref_w = panels[names[0]].shape[:2]
    target_w, target_h = int(ref_w * scale), int(ref_h * scale)

    labeled_panels = []
    for name in names:
        img = panels[name]
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        labeled_panels.append(add_label(resized, name))

    # Pad the panel list to a multiple of 2 for a clean grid
    while len(labeled_panels) % 2 != 0:
        blank = np.zeros_like(labeled_panels[0])
        labeled_panels.append(blank)

    rows = []
    for i in range(0, len(labeled_panels), 2):
        rows.append(np.hstack(labeled_panels[i:i + 2]))

    return np.vstack(rows)

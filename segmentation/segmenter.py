"""
segmenter.py
------------
Wraps MediaPipe Selfie Segmentation to produce a pixel-level person/background
mask each frame: a binary mask, a soft alpha mask, and a background-removed
RGBA/BGR composite. Includes temporal smoothing (EMA) so the mask edges don't
flicker frame to frame, which matters for stable garment layering later.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

import config


@dataclass
class SegmentationResult:
    """Container for a single frame's segmentation output."""

    success: bool
    alpha_mask: Optional[np.ndarray] = None    # float32, HxW, values in [0, 1]
    binary_mask: Optional[np.ndarray] = None   # uint8, HxW, values in {0, 255}


class PersonSegmenter:
    """Real-time person segmenter built on top of MediaPipe Selfie Segmentation."""

    def __init__(self):
        self._mp_selfie = mp.solutions.selfie_segmentation
        try:
            self._segmenter = self._mp_selfie.SelfieSegmentation(
                model_selection=config.SEGMENTATION_MODEL_SELECTION
            )
        except Exception as exc:  # pragma: no cover - defensive init guard
            raise RuntimeError(
                f"Failed to initialize MediaPipe Selfie Segmentation model: {exc}"
            ) from exc

        self._prev_mask: Optional[np.ndarray] = None  # for temporal smoothing

    def process(self, frame_bgr: np.ndarray) -> SegmentationResult:
        """
        Run segmentation on a single BGR frame.

        Args:
            frame_bgr: Input frame in BGR order.

        Returns:
            SegmentationResult with alpha (soft) and binary (hard) masks.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return SegmentationResult(success=False)

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self._segmenter.process(frame_rgb)
        except Exception as exc:
            print(f"[PersonSegmenter] Inference error: {exc}")
            return SegmentationResult(success=False)

        if results.segmentation_mask is None:
            return SegmentationResult(success=False)

        raw_mask = results.segmentation_mask.astype(np.float32)

        # Temporal smoothing (exponential moving average) to reduce flicker
        alpha = config.SEGMENTATION_TEMPORAL_SMOOTHING
        if self._prev_mask is not None and self._prev_mask.shape == raw_mask.shape:
            smoothed = alpha * self._prev_mask + (1.0 - alpha) * raw_mask
        else:
            smoothed = raw_mask
        self._prev_mask = smoothed

        binary_mask = (smoothed > config.SEGMENTATION_THRESHOLD).astype(np.uint8) * 255

        return SegmentationResult(success=True, alpha_mask=smoothed, binary_mask=binary_mask)

    @staticmethod
    def remove_background(
        frame_bgr: np.ndarray,
        result: SegmentationResult,
        background_color=(0, 0, 0),
    ) -> np.ndarray:
        """
        Composite the person over a solid background color using the soft alpha mask.

        Args:
            frame_bgr: Original frame.
            result: Output of `process()`.
            background_color: BGR color to use where the mask says "background".

        Returns:
            Composited BGR image the same size as the input frame.
        """
        if not result.success or result.alpha_mask is None:
            return frame_bgr.copy()

        alpha_3ch = np.repeat(result.alpha_mask[:, :, np.newaxis], 3, axis=2)
        bg = np.full_like(frame_bgr, background_color, dtype=np.uint8)
        composite = (frame_bgr.astype(np.float32) * alpha_3ch
                     + bg.astype(np.float32) * (1.0 - alpha_3ch))
        return np.clip(composite, 0, 255).astype(np.uint8)

    @staticmethod
    def overlay_mask(
        frame_bgr: np.ndarray,
        result: SegmentationResult,
        color=None,
        alpha_strength: float = None,
    ) -> np.ndarray:
        """
        Draw a translucent color overlay wherever the person mask is active.

        Args:
            frame_bgr: Original frame.
            result: Output of `process()`.
            color: BGR overlay color (defaults to config.MASK_OVERLAY_COLOR).
            alpha_strength: Overlay opacity in [0, 1] (defaults to config.MASK_OVERLAY_ALPHA).

        Returns:
            Annotated copy of the frame.
        """
        if not result.success or result.binary_mask is None:
            return frame_bgr.copy()

        color = color or config.MASK_OVERLAY_COLOR
        alpha_strength = config.MASK_OVERLAY_ALPHA if alpha_strength is None else alpha_strength

        overlay = frame_bgr.copy()
        colored_layer = np.full_like(frame_bgr, color, dtype=np.uint8)
        mask_bool = result.binary_mask > 0
        overlay[mask_bool] = cv2.addWeighted(
            frame_bgr, 1 - alpha_strength, colored_layer, alpha_strength, 0
        )[mask_bool]
        return overlay

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._segmenter.close()
        except Exception as exc:
            print(f"[PersonSegmenter] Error while closing: {exc}")

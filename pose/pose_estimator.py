"""
pose_estimator.py
-----------------
Wraps MediaPipe Pose to provide real-time body-landmark detection with
x, y, z and visibility (confidence) for 33 body joints, plus metric-scale
"world landmarks" that are useful as depth anchors for the mesh stage.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

import config


@dataclass
class PoseLandmark:
    """A single body landmark."""

    index: int
    name: str
    x: float          # Normalized image x in [0, 1]
    y: float          # Normalized image y in [0, 1]
    z: float          # Depth relative to hips (roughly same scale as x); negative = closer to camera
    visibility: float  # Confidence in [0, 1] that the landmark is visible
    pixel_x: int = 0   # Pixel-space x (filled in once frame size is known)
    pixel_y: int = 0   # Pixel-space y


@dataclass
class PoseResult:
    """Container for a single frame's pose estimation output."""

    detected: bool
    landmarks: List[PoseLandmark] = field(default_factory=list)
    world_landmarks: Optional[np.ndarray] = None  # (33, 3) metric-scale coordinates, hip-centered

    def get_landmark(self, name: str) -> Optional[PoseLandmark]:
        """Look up a landmark by its MediaPipe name (e.g. 'LEFT_SHOULDER')."""
        for lm in self.landmarks:
            if lm.name == name:
                return lm
        return None

    def as_pixel_array(self) -> np.ndarray:
        """Return an (N, 3) array of [pixel_x, pixel_y, z] for all landmarks."""
        if not self.landmarks:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(
            [[lm.pixel_x, lm.pixel_y, lm.z] for lm in self.landmarks],
            dtype=np.float32,
        )


class PoseEstimator:
    """Real-time body pose estimator built on top of MediaPipe Pose."""

    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        try:
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=config.POSE_MODEL_COMPLEXITY,
                smooth_landmarks=config.POSE_SMOOTH_LANDMARKS,
                enable_segmentation=config.POSE_ENABLE_SEGMENTATION,
                min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
            )
        except Exception as exc:  # pragma: no cover - defensive init guard
            raise RuntimeError(
                f"Failed to initialize MediaPipe Pose model: {exc}"
            ) from exc

        # Precompute landmark names in index order (PoseLandmark enum -> name)
        self._landmark_names = [lm.name for lm in self._mp_pose.PoseLandmark]
        self.connections = self._mp_pose.POSE_CONNECTIONS

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        """
        Run pose estimation on a single BGR frame.

        Args:
            frame_bgr: The input frame in OpenCV's default BGR color order.

        Returns:
            PoseResult with per-landmark x, y, z, visibility and pixel coords.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return PoseResult(detected=False)

        h, w = frame_bgr.shape[:2]

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self._pose.process(frame_rgb)
        except Exception as exc:
            print(f"[PoseEstimator] Inference error: {exc}")
            return PoseResult(detected=False)

        if not results.pose_landmarks:
            return PoseResult(detected=False)

        landmarks: List[PoseLandmark] = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks.append(
                PoseLandmark(
                    index=idx,
                    name=self._landmark_names[idx],
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility,
                    pixel_x=int(np.clip(lm.x * w, 0, w - 1)),
                    pixel_y=int(np.clip(lm.y * h, 0, h - 1)),
                )
            )

        world_landmarks = None
        if results.pose_world_landmarks:
            world_landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark],
                dtype=np.float32,
            )

        return PoseResult(detected=True, landmarks=landmarks, world_landmarks=world_landmarks)

    def draw(self, frame_bgr: np.ndarray, result: PoseResult) -> np.ndarray:
        """
        Draw the skeleton (joints + connecting bones) onto a copy of the frame.

        Args:
            frame_bgr: Frame to draw on (a copy is made; original is untouched).
            result: Output of `process()`.

        Returns:
            Annotated copy of the frame.
        """
        annotated = frame_bgr.copy()
        if not result.detected:
            return annotated

        points = {lm.index: (lm.pixel_x, lm.pixel_y) for lm in result.landmarks}
        visibility = {lm.index: lm.visibility for lm in result.landmarks}
        min_vis = config.POSE_MIN_DETECTION_CONFIDENCE

        # Draw bones
        for start_idx, end_idx in self.connections:
            if start_idx not in points or end_idx not in points:
                continue
            if visibility[start_idx] < min_vis or visibility[end_idx] < min_vis:
                continue
            cv2.line(annotated, points[start_idx], points[end_idx], config.COLOR_SKELETON, 2, cv2.LINE_AA)

        # Draw joints
        for idx, (px, py) in points.items():
            if visibility[idx] < min_vis:
                continue
            cv2.circle(annotated, (px, py), 4, config.COLOR_JOINT, -1, cv2.LINE_AA)

        return annotated

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._pose.close()
        except Exception as exc:
            print(f"[PoseEstimator] Error while closing: {exc}")

"""
config.py
---------
Centralized configuration for the Stage 1 (Body Understanding) pipeline.

Every tunable parameter used across pose estimation, segmentation, mesh
building, visualization and I/O lives here so nothing is hardcoded deep
inside the pipeline modules.
"""

import os

# ---------------------------------------------------------------------------
# Camera / capture settings
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0            # 0 = default webcam. Change to 1, 2... for external cams.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FLIP_HORIZONTAL = True       # Mirror the feed so it behaves like a mirror/selfie cam

# ---------------------------------------------------------------------------
# Pose estimation (MediaPipe Pose)
# ---------------------------------------------------------------------------
POSE_MODEL_COMPLEXITY = 1            # 0 = lite (fastest), 1 = full, 2 = heavy (most accurate)
POSE_MIN_DETECTION_CONFIDENCE = 0.6
POSE_MIN_TRACKING_CONFIDENCE = 0.6
POSE_SMOOTH_LANDMARKS = True
POSE_ENABLE_SEGMENTATION = False     # We use a dedicated, tunable segmentation model instead

# ---------------------------------------------------------------------------
# Segmentation (MediaPipe Selfie Segmentation)
# ---------------------------------------------------------------------------
SEGMENTATION_MODEL_SELECTION = 1     # 0 = general model, 1 = landscape model (fast, full body)
SEGMENTATION_THRESHOLD = 0.5         # Probability threshold used to binarize the raw mask
SEGMENTATION_TEMPORAL_SMOOTHING = 0.6  # EMA factor in [0,1]; higher = smoother but laggier

# ---------------------------------------------------------------------------
# Mesh estimation
# ---------------------------------------------------------------------------
# Stage 1 does not (yet) fit a parametric body model such as SMPL. Instead it
# builds a dense triangulated 2.5D mesh over the segmented body silhouette,
# using the MediaPipe pose landmarks as anchor/key vertices and depth (z)
# values interpolated from the pose landmarks. This mesh already has the
# right *shape* (vertices + triangle faces + per-vertex depth) required by a
# later garment-warping stage; swapping it for an SMPL mesh later only
# requires changing MeshEstimator's internals, not its output interface.
MESH_GRID_STEP = 18                  # Pixel spacing between interior sample points
MESH_MAX_INTERIOR_POINTS = 900       # Safety cap to keep triangulation real-time
MESH_DEPTH_SCALE = 220.0             # Scales normalized landmark z into visible pixel-ish units

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
COLOR_SKELETON = (0, 255, 0)
COLOR_JOINT = (0, 0, 255)
COLOR_MESH_EDGE = (255, 200, 0)
COLOR_MESH_VERTEX = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_FPS_BG = (0, 0, 0)
COLOR_SAVE_FLASH = (0, 255, 255)
MASK_OVERLAY_ALPHA = 0.45
MASK_OVERLAY_COLOR = (0, 255, 0)
PANEL_LABEL_HEIGHT = 28
DISPLAY_SCALE = 0.5                  # Scale factor applied to each panel in the 2x2 grid

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ---------------------------------------------------------------------------
# Keybindings
# ---------------------------------------------------------------------------
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")

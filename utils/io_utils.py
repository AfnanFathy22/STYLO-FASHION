"""
io_utils.py
-----------
Handles all disk I/O: creating timestamped output folders and saving every
artifact produced by a single capture (frame, pose overlay, masks,
background-removed image, and the mesh as both a raw .npz and a standard
.obj file that can be opened in Blender/MeshLab for inspection).
"""

import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import config
from mesh.mesh_estimator import BodyMesh
from pose.pose_estimator import PoseResult


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it doesn't already exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Could not create output directory '{path}': {exc}") from exc


def save_obj(mesh: BodyMesh, path: str) -> None:
    """Write a BodyMesh out as a Wavefront .obj file (viewable in Blender/MeshLab)."""
    if not mesh.valid:
        raise ValueError("Cannot save an invalid/empty mesh")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Body mesh exported from Stage 1 (Body Understanding)\n")
        f.write(f"# vertices: {len(mesh.vertices)}, faces: {len(mesh.triangles)}\n")
        for v in mesh.vertices:
            # Flip y and negate z so the mesh looks upright in standard 3D viewers
            f.write(f"v {v[0]:.4f} {-v[1]:.4f} {-v[2]:.4f}\n")
        for tri in mesh.triangles:
            # OBJ face indices are 1-based
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def save_capture(
    original_frame: np.ndarray,
    pose_overlay: np.ndarray,
    segmentation_overlay: np.ndarray,
    binary_mask: Optional[np.ndarray],
    alpha_mask: Optional[np.ndarray],
    background_removed: Optional[np.ndarray],
    mesh_overlay: np.ndarray,
    mesh: BodyMesh,
    pose_result: PoseResult,
    output_root: str = None,
) -> str:
    """
    Save every artifact from the current frame into a fresh timestamped folder.

    Returns:
        The path to the created output folder.
    """
    output_root = output_root or config.OUTPUT_DIR
    ensure_dir(output_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    capture_dir = os.path.join(output_root, f"capture_{timestamp}")
    ensure_dir(capture_dir)

    try:
        cv2.imwrite(os.path.join(capture_dir, "original.png"), original_frame)
        cv2.imwrite(os.path.join(capture_dir, "pose_overlay.png"), pose_overlay)
        cv2.imwrite(os.path.join(capture_dir, "segmentation_overlay.png"), segmentation_overlay)
        cv2.imwrite(os.path.join(capture_dir, "mesh_overlay.png"), mesh_overlay)

        if binary_mask is not None:
            cv2.imwrite(os.path.join(capture_dir, "binary_mask.png"), binary_mask)

        if alpha_mask is not None:
            alpha_vis = (np.clip(alpha_mask, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(capture_dir, "alpha_mask.png"), alpha_vis)

        if background_removed is not None:
            cv2.imwrite(os.path.join(capture_dir, "background_removed.png"), background_removed)

        # Save raw pose landmarks as a simple text/CSV-style file for inspection
        landmarks_path = os.path.join(capture_dir, "pose_landmarks.csv")
        with open(landmarks_path, "w", encoding="utf-8") as f:
            f.write("index,name,x,y,z,visibility,pixel_x,pixel_y\n")
            for lm in pose_result.landmarks:
                f.write(
                    f"{lm.index},{lm.name},{lm.x:.6f},{lm.y:.6f},{lm.z:.6f},"
                    f"{lm.visibility:.4f},{lm.pixel_x},{lm.pixel_y}\n"
                )

        # Save mesh as both a fast-to-reload .npz and a viewer-friendly .obj
        if mesh.valid:
            np.savez(
                os.path.join(capture_dir, "mesh.npz"),
                vertices=mesh.vertices,
                triangles=mesh.triangles,
            )
            save_obj(mesh, os.path.join(capture_dir, "mesh.obj"))

    except Exception as exc:
        raise RuntimeError(f"Failed while saving capture to '{capture_dir}': {exc}") from exc

    return capture_dir

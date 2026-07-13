"""
mesh_estimator.py
-----------------
Builds a dense, triangulated 2.5D body mesh suitable as an input to a future
garment-warping stage (Stage 3+: CatVTON / IDM-VTON style diffusion warping,
or a classic UV-mapping warp).

Why not SMPL directly in Stage 1?
    A full parametric body model (SMPL/SMPL-X) needs its own fitting
    procedure (shape + pose betas optimized against 2D/3D keypoints) and a
    bundled statistical body model file, which is a heavier dependency than
    is appropriate for a first "body understanding" stage. Instead, this
    module builds a mesh directly from what Stage 1 already has for free:

        1. The person segmentation mask -> defines *where* the body surface is
           (including concavities like the space between the arm and torso,
           or between the legs), which a bounding box or convex hull would not.
        2. The MediaPipe pose landmarks (with x, y, z) -> anchor/key vertices
           with real depth estimates, used to interpolate depth for every
           other mesh vertex.

    The output (vertices + triangle faces + landmark anchor indices) has
    exactly the shape a warping stage needs. Swapping this for SMPL later
    only requires changing what populates `BodyMesh`, not how downstream
    code consumes it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

import config

try:
    from scipy.spatial import Delaunay, cKDTree
except ImportError as exc:  # pragma: no cover - surfaced clearly at runtime
    raise ImportError(
        "scipy is required for mesh triangulation. Install it with "
        "'pip install scipy' (see requirements.txt)."
    ) from exc

from pose.pose_estimator import PoseResult


@dataclass
class BodyMesh:
    """A triangulated body mesh anchored to pose landmarks."""

    vertices: np.ndarray                      # (N, 3) float32: [pixel_x, pixel_y, z]
    triangles: np.ndarray                     # (M, 3) int32: vertex indices per triangle
    landmark_vertex_indices: Dict[str, int] = field(default_factory=dict)
    valid: bool = False


class MeshEstimator:
    """Builds a per-frame body mesh from a segmentation mask + pose landmarks."""

    def __init__(
        self,
        grid_step: int = None,
        max_interior_points: int = None,
        depth_scale: float = None,
    ):
        self.grid_step = grid_step or config.MESH_GRID_STEP
        self.max_interior_points = max_interior_points or config.MESH_MAX_INTERIOR_POINTS
        self.depth_scale = depth_scale or config.MESH_DEPTH_SCALE

    def build(self, binary_mask: Optional[np.ndarray], pose_result: PoseResult) -> BodyMesh:
        """
        Build a body mesh for the current frame.

        Args:
            binary_mask: uint8 HxW mask (0 or 255) from PersonSegmenter. If None
                or empty, only pose landmarks are used as a sparse point cloud.
            pose_result: Output of PoseEstimator.process().

        Returns:
            BodyMesh. `valid` is False if there weren't enough points to
            triangulate (e.g. no person detected).
        """
        if not pose_result.detected or not pose_result.landmarks:
            return BodyMesh(vertices=np.empty((0, 3), dtype=np.float32),
                             triangles=np.empty((0, 3), dtype=np.int32),
                             valid=False)

        try:
            landmark_points, landmark_names = self._landmark_points(pose_result)
            interior_points = self._sample_interior_points(binary_mask, landmark_points)

            all_points_2d = np.vstack([landmark_points, interior_points]) \
                if len(interior_points) > 0 else landmark_points

            if len(all_points_2d) < 4:
                # Delaunay triangulation needs at least a handful of non-collinear points
                return BodyMesh(vertices=np.empty((0, 3), dtype=np.float32),
                                 triangles=np.empty((0, 3), dtype=np.int32),
                                 valid=False)

            depths = self._interpolate_depth(all_points_2d, landmark_points, pose_result)
            vertices = np.hstack([all_points_2d, depths.reshape(-1, 1)]).astype(np.float32)

            triangles = self._triangulate(all_points_2d, binary_mask)

            landmark_vertex_indices = {
                name: idx for idx, name in enumerate(landmark_names)
            }

            return BodyMesh(
                vertices=vertices,
                triangles=triangles,
                landmark_vertex_indices=landmark_vertex_indices,
                valid=len(triangles) > 0,
            )

        except Exception as exc:
            print(f"[MeshEstimator] Failed to build mesh: {exc}")
            return BodyMesh(vertices=np.empty((0, 3), dtype=np.float32),
                             triangles=np.empty((0, 3), dtype=np.int32),
                             valid=False)

    def _landmark_points(self, pose_result: PoseResult):
        """Extract visible landmark pixel coordinates as anchor mesh vertices."""
        points = []
        names = []
        for lm in pose_result.landmarks:
            if lm.visibility < config.POSE_MIN_DETECTION_CONFIDENCE:
                continue
            points.append([lm.pixel_x, lm.pixel_y])
            names.append(lm.name)
        return np.array(points, dtype=np.float32), names

    def _sample_interior_points(
        self, binary_mask: Optional[np.ndarray], landmark_points: np.ndarray
    ) -> np.ndarray:
        """Sample a grid of points inside the body silhouette to densify the mesh."""
        if binary_mask is None or binary_mask.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        h, w = binary_mask.shape[:2]
        ys, xs = np.mgrid[0:h:self.grid_step, 0:w:self.grid_step]
        candidates = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

        # Keep only points that fall inside the segmented person
        mask_vals = binary_mask[
            np.clip(candidates[:, 1].astype(int), 0, h - 1),
            np.clip(candidates[:, 0].astype(int), 0, w - 1),
        ]
        interior = candidates[mask_vals > 0]

        # Avoid an explosion of points on large/close subjects (real-time safety cap)
        if len(interior) > self.max_interior_points:
            rng = np.random.default_rng(seed=42)
            keep_idx = rng.choice(len(interior), size=self.max_interior_points, replace=False)
            interior = interior[keep_idx]

        return interior

    def _interpolate_depth(
        self, all_points_2d: np.ndarray, landmark_points: np.ndarray, pose_result: PoseResult
    ) -> np.ndarray:
        """
        Assign a depth (z) value to every mesh vertex by inverse-distance
        weighting from the nearest visible pose landmarks (which already
        carry a MediaPipe-estimated z).
        """
        landmark_z = np.array(
            [lm.z for lm in pose_result.landmarks if lm.visibility >= config.POSE_MIN_DETECTION_CONFIDENCE],
            dtype=np.float32,
        )
        if len(landmark_z) == 0:
            return np.zeros(len(all_points_2d), dtype=np.float32)

        tree = cKDTree(landmark_points)
        k = min(4, len(landmark_points))
        distances, indices = tree.query(all_points_2d, k=k)

        # query() drops the k-dimension when k == 1; normalize shape
        if k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        weights = 1.0 / np.maximum(distances, 1e-3)
        weights /= weights.sum(axis=1, keepdims=True)
        depths = np.sum(weights * landmark_z[indices], axis=1)

        return depths * self.depth_scale

    def _triangulate(self, points_2d: np.ndarray, binary_mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Run Delaunay triangulation and drop any triangle whose centroid falls
        outside the person mask, so the mesh respects concave silhouette
        boundaries (armpits, between the legs, etc.) instead of filling in
        the convex hull.
        """
        delaunay = Delaunay(points_2d)
        triangles = delaunay.simplices

        if binary_mask is None or binary_mask.size == 0:
            return triangles.astype(np.int32)

        h, w = binary_mask.shape[:2]
        centroids = points_2d[triangles].mean(axis=1)
        cx = np.clip(centroids[:, 0].astype(int), 0, w - 1)
        cy = np.clip(centroids[:, 1].astype(int), 0, h - 1)
        keep = binary_mask[cy, cx] > 0

        return triangles[keep].astype(np.int32)

    @staticmethod
    def draw(frame_bgr: np.ndarray, mesh: BodyMesh) -> np.ndarray:
        """
        Draw the mesh wireframe (triangle edges + vertices) onto a copy of the frame.
        """
        annotated = frame_bgr.copy()
        if not mesh.valid or len(mesh.triangles) == 0:
            return annotated

        pts = mesh.vertices[:, :2].astype(int)

        for tri in mesh.triangles:
            p0, p1, p2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            cv2.line(annotated, tuple(p0), tuple(p1), config.COLOR_MESH_EDGE, 1, cv2.LINE_AA)
            cv2.line(annotated, tuple(p1), tuple(p2), config.COLOR_MESH_EDGE, 1, cv2.LINE_AA)
            cv2.line(annotated, tuple(p2), tuple(p0), config.COLOR_MESH_EDGE, 1, cv2.LINE_AA)

        for p in pts:
            cv2.circle(annotated, tuple(p), 2, config.COLOR_MESH_VERTEX, -1, cv2.LINE_AA)

        return annotated

"""
run_image.py
------------
Standalone Stage 1 (Body Understanding) runner for a SINGLE IMAGE FILE.

Use this instead of main.py when you don't have a working webcam, or you
just want to test/inspect the pipeline (pose + segmentation + mesh) on one
specific photo. It never touches the camera.

Run with:
    python run_image.py --image path/to/photo.jpg

Optional:
    python run_image.py --image path/to/photo.jpg --output outputs/my_test

Controls in the result window:
    S  -> save every artifact (overlays, masks, mesh.obj, landmarks CSV) to disk
    Q  -> close the window and exit
    (any other key also just closes the window without saving)
"""

import argparse
import os
import sys

import cv2

import config
from pose.pose_estimator import PoseEstimator
from segmentation.segmenter import PersonSegmenter
from mesh.mesh_estimator import MeshEstimator
from utils import visualization, io_utils


WINDOW_NAME = "Stage 1 - Body Understanding [Image Mode] (S=save, Q/any key=quit)"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Stage 1 (pose + segmentation + mesh) on a single image file."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file (jpg, png, bmp, ...).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output folder root (defaults to config.OUTPUT_DIR / 'outputs').",
    )
    return parser.parse_args()


def load_image(image_path: str):
    """Load an image from disk with clear, actionable error messages."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"No file found at '{image_path}'. Check the path (use an absolute path "
            "if you're not sure the terminal's current folder matches)."
        )

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(
            f"'{image_path}' exists but could not be read as an image. "
            "Make sure it's a valid, non-corrupted jpg/png/bmp file."
        )

    return frame


def main() -> int:
    """Run the Stage 1 pipeline once on a single image. Returns a process exit code."""
    args = parse_args()

    print("=" * 70)
    print("Stage 1: Body Understanding - Image Mode")
    print("=" * 70)

    try:
        frame = load_image(args.image)
    except Exception as exc:
        print(f"[FATAL] {exc}")
        return 1

    print(f"Loaded image: {args.image}  (size: {frame.shape[1]}x{frame.shape[0]})")
    print("Initializing models (this can take a few seconds on first run)...")

    try:
        pose_estimator = PoseEstimator()
        segmenter = PersonSegmenter()
        mesh_estimator = MeshEstimator()
    except Exception as exc:
        print(f"[FATAL] Failed to initialize models: {exc}")
        return 1

    try:
        # --- Stage 1 core processing -------------------------------------------------------
        try:
            pose_result = pose_estimator.process(frame)
        except Exception as exc:
            print(f"[FATAL] Pose estimation failed: {exc}")
            return 1

        try:
            seg_result = segmenter.process(frame)
        except Exception as exc:
            print(f"[ERROR] Segmentation failed: {exc}")
            seg_result = None

        try:
            mesh = mesh_estimator.build(
                seg_result.binary_mask if seg_result and seg_result.success else None,
                pose_result,
            )
        except Exception as exc:
            print(f"[ERROR] Mesh estimation failed: {exc}")
            mesh = None

        print(f"Person detected: {'yes' if pose_result.detected else 'no'}")
        if not pose_result.detected:
            print(
                "[NOTE] No pose was detected in this image. The panels below will still "
                "display, just without a skeleton/mesh overlay. Try a photo where a full "
                "or partial body is clearly visible against a reasonably plain background."
            )

        # --- Visualization -------------------------------------------------------------------
        pose_overlay = pose_estimator.draw(frame, pose_result)

        if seg_result and seg_result.success:
            segmentation_overlay = segmenter.overlay_mask(frame, seg_result)
            background_removed = segmenter.remove_background(frame, seg_result)
        else:
            segmentation_overlay = frame.copy()
            background_removed = frame.copy()

        if mesh and mesh.valid:
            mesh_overlay = mesh_estimator.draw(frame, mesh)
        else:
            mesh_overlay = frame.copy()

        panels = {
            "Pose": pose_overlay,
            "Segmentation": segmentation_overlay,
            "Mesh": mesh_overlay,
            "Background Removed": background_removed,
        }
        display = visualization.make_panel_grid(panels)

        print("Showing result window. Press 'S' to save all outputs, any other key to exit.")
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(0) & 0xFF

        if key == config.KEY_SAVE:
            if mesh is None:
                mesh = mesh_estimator.build(None, pose_result)
            try:
                save_dir = io_utils.save_capture(
                    original_frame=frame,
                    pose_overlay=pose_overlay,
                    segmentation_overlay=segmentation_overlay,
                    binary_mask=seg_result.binary_mask if seg_result and seg_result.success else None,
                    alpha_mask=seg_result.alpha_mask if seg_result and seg_result.success else None,
                    background_removed=background_removed,
                    mesh_overlay=mesh_overlay,
                    mesh=mesh,
                    pose_result=pose_result,
                    output_root=args.output,
                )
                print(f"[SAVED] Capture written to: {save_dir}")
            except Exception as exc:
                print(f"[ERROR] Could not save capture: {exc}")
        else:
            print("Exiting without saving (press 'S' next time if you want to keep the outputs).")

    finally:
        cv2.destroyAllWindows()
        pose_estimator.close()
        segmenter.close()
        print("All resources released. Goodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

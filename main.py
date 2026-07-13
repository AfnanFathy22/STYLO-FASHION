"""
main.py
-------
Stage 1 (Body Understanding) of the AI Virtual Try-On pipeline.

Runs a live webcam loop that, per frame:
    1. Estimates body pose (33 landmarks with x, y, z, visibility)
    2. Segments the person from the background (binary + soft alpha mask)
    3. Builds a triangulated body mesh anchored to the pose landmarks,
       suitable as input to a future garment-warping stage
    4. Visualizes all three outputs plus an FPS counter in one window
    5. Lets the user save every artifact of the current frame to disk (S)
       or quit the application (Q)

Run with:  python main.py
"""

import sys
import time

import cv2

import config
from pose.pose_estimator import PoseEstimator
from segmentation.segmenter import PersonSegmenter
from mesh.mesh_estimator import MeshEstimator
from utils.fps_counter import FPSCounter
from utils import visualization, io_utils


WINDOW_NAME = "Stage 1 - Body Understanding (Q=quit, S=save)"
SAVE_FLASH_DURATION_FRAMES = 15  # roughly half a second at ~30 FPS


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open the webcam with error handling and configure its resolution."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {camera_index}. "
            "Check that a webcam is connected, not in use by another app, "
            "and that camera permissions are granted to your terminal/IDE."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def main() -> int:
    """Run the Stage 1 live pipeline. Returns a process exit code."""
    print("=" * 70)
    print("Stage 1: Body Understanding - AI Virtual Try-On Pipeline")
    print("=" * 70)
    print("Initializing models (this can take a few seconds on first run)...")

    try:
        pose_estimator = PoseEstimator()
        segmenter = PersonSegmenter()
        mesh_estimator = MeshEstimator()
    except Exception as exc:
        print(f"[FATAL] Failed to initialize models: {exc}")
        return 1

    try:
        cap = open_camera(config.CAMERA_INDEX)
    except Exception as exc:
        print(f"[FATAL] {exc}")
        pose_estimator.close()
        segmenter.close()
        return 1

    fps_counter = FPSCounter(window_size=30)
    save_flash_countdown = 0
    last_status_print = 0.0

    print("Camera opened successfully. Press 'S' to save a capture, 'Q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Failed to read frame from camera. Retrying...")
                time.sleep(0.05)
                continue

            if config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # --- Stage 1 core processing -------------------------------------------------
            try:
                pose_result = pose_estimator.process(frame)
            except Exception as exc:
                print(f"[ERROR] Pose estimation failed on this frame: {exc}")
                continue

            try:
                seg_result = segmenter.process(frame)
            except Exception as exc:
                print(f"[ERROR] Segmentation failed on this frame: {exc}")
                seg_result = None

            try:
                mesh = mesh_estimator.build(
                    seg_result.binary_mask if seg_result and seg_result.success else None,
                    pose_result,
                )
            except Exception as exc:
                print(f"[ERROR] Mesh estimation failed on this frame: {exc}")
                mesh = None

            # --- Visualization -------------------------------------------------------------
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

            fps_counter.tick()
            fps = fps_counter.get_fps()
            visualization.draw_fps(pose_overlay, fps)

            panels = {
                "Pose": pose_overlay,
                "Segmentation": segmentation_overlay,
                "Mesh": mesh_overlay,
                "Background Removed": background_removed,
            }
            display = visualization.make_panel_grid(panels)

            if save_flash_countdown > 0:
                display = visualization.draw_save_flash(display, "SAVED")
                save_flash_countdown -= 1

            cv2.imshow(WINDOW_NAME, display)

            # Periodic console heartbeat (useful when running headless-ish / logging)
            now = time.time()
            if now - last_status_print > 5.0:
                detected = "yes" if pose_result.detected else "no"
                print(f"[STATUS] FPS={fps:.1f} | person_detected={detected}")
                last_status_print = now

            # --- Key handling ---------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == config.KEY_QUIT:
                print("Quit key pressed. Shutting down...")
                break

            elif key == config.KEY_SAVE:
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
                    )
                    print(f"[SAVED] Capture written to: {save_dir}")
                    save_flash_countdown = SAVE_FLASH_DURATION_FRAMES
                except Exception as exc:
                    print(f"[ERROR] Could not save capture: {exc}")

            # Also allow closing via the window's [x] button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user. Shutting down...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Shutting down...")
    except Exception as exc:
        print(f"[FATAL] Unexpected error in main loop: {exc}")
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_estimator.close()
        segmenter.close()
        print("All resources released. Goodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

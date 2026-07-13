"""
process_video_tryon.py
-----------------------
Runs virtual try-on on EVERY (or every Nth) frame of a video using a LOCAL
model (idm or ootd, needs GPU), and writes a new video with the garment
swapped in.

Why not the remote HF Space for this: the free 'idm_remote' Space has a tiny
daily ZeroGPU quota (~25 min/day, ~60s/call) and is way too slow/limited to
process hundreds of frames. A local GPU model has no such quota.

Usage:
    python process_video_tryon.py --video video1_fixed.mp4 --garment garment.jpeg --model ootd

Useful flags:
    --process-every N   only run try-on every N-th frame, and reuse the last
                         try-on result for frames in between (default: 1 =
                         every frame -- slow but smoothest result). Raise this
                         (e.g. 5 or 10) to finish much faster at the cost of
                         some frames looking slightly "stepped" instead of
                         perfectly smooth.
    --garment-desc TEXT  short text description of the garment (helps some
                         models, e.g. "a black dress"). Optional.
    --output PATH        output video path (default: <video>_tryon.mp4)
    --keep-audio         copy the original video's audio track onto the
                         output using ffmpeg (requires ffmpeg on PATH).
"""

import argparse
import os
import subprocess
import sys
import time

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Full-video virtual try-on (local GPU model).")
    p.add_argument("--video", required=True, help="Input video path.")
    p.add_argument("--garment", required=True, help="Garment image path.")
    p.add_argument("--garment-desc", default=None, help="Optional garment text description.")
    p.add_argument("--model", choices=["idm", "ootd"], default="ootd",
                    help="Local try-on engine to use (default: ootd -- easier to set up than idm).")
    p.add_argument("--process-every", type=int, default=1,
                    help="Run try-on every N-th frame; frames in between reuse the last result. Default: 1.")
    p.add_argument("--output", default=None, help="Output video path (default: <video>_tryon.mp4).")
    p.add_argument("--keep-audio", action="store_true", help="Copy original audio onto output via ffmpeg.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.model == "ootd":
        from tryon.ootd_wrapper import run_tryon
    else:
        from tryon.idm_vton_wrapper import run_tryon

    if not os.path.exists(args.video):
        print(f"[FATAL] Video not found: {args.video}")
        return 1
    if not os.path.exists(args.garment):
        print(f"[FATAL] Garment image not found: {args.garment}")
        return 1

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[FATAL] Could not open video: {args.video}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_path = args.output or (os.path.splitext(args.video)[0] + "_tryon.mp4")
    silent_path = out_path if not args.keep_audio else out_path + ".silent.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (width, height))

    print(f"Video: {args.video}  ({width}x{height}, {fps:.1f} fps, {total_frames} frames)")
    print(f"Model: {args.model}   process-every: {args.process_every}")
    print(f"Output: {out_path}")
    print("Starting... this can take a while for long videos.")

    last_result = None
    frame_index = -1
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1

        run_now = (frame_index % max(args.process_every, 1)) == 0 or last_result is None

        if run_now:
            try:
                t0 = time.time()
                kwargs = dict(person_image=frame, garment_image_path=args.garment)
                if args.garment_desc:
                    kwargs["garment_description"] = args.garment_desc
                result = run_tryon(**kwargs)
                last_result = result
                print(f"[FRAME {frame_index}/{total_frames}] try-on done in {time.time()-t0:.1f}s")
            except Exception as exc:
                print(f"[FRAME {frame_index}] try-on FAILED: {exc} -- using original frame")
                last_result = frame
        else:
            print(f"[FRAME {frame_index}/{total_frames}] reusing last try-on result")

        out_frame = last_result
        if out_frame.shape[:2] != (height, width):
            out_frame = cv2.resize(out_frame, (width, height))
        writer.write(out_frame)

    cap.release()
    writer.release()
    print(f"Done. Silent video written to: {silent_path}  (elapsed {time.time()-t_start:.1f}s)")

    if args.keep_audio:
        print("Merging original audio with ffmpeg...")
        cmd = [
            "ffmpeg", "-y",
            "-i", silent_path,
            "-i", args.video,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True)
            os.remove(silent_path)
            print(f"Final video with audio: {out_path}")
        except Exception as exc:
            print(f"[WARNING] Could not merge audio ({exc}). Silent video kept at: {silent_path}")

    print(f"Final output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
batch_tryon.py
---------------
Run virtual try-on on MULTIPLE person/garment image pairs in one go,
instead of testing one image at a time (like test_remote_tryon.py does).

Two modes:

1) FOLDER MODE (all combinations):
   Put several person photos in one folder and several garment photos in
   another folder. This script will try every person x every garment
   combination (or just matching pairs, see --pairing).

       python batch_tryon.py --persons_dir persons --garments_dir garments

2) MANIFEST MODE (explicit control):
   Create a simple CSV file listing exact pairs + descriptions you want to
   test, for precise control over what gets tried:

       person,garment,description
       persons/p1.jpg,garments/g1.jpg,a black dress
       persons/p2.jpg,garments/g2.jpg,a red hoodie

       python batch_tryon.py --manifest pairs.csv

Every result is saved to outputs/batch_results/ with a clear filename, and
a summary log (results_log.csv) is written so you can open it in Excel and
fill in a manual 1-5 quality score for your graduation project write-up.
"""

import argparse
import csv
import os
import sys
import time

import cv2

from tryon.idm_vton_remote import run_tryon


RESULTS_DIR = os.path.join("outputs", "batch_results")
LOG_PATH = os.path.join(RESULTS_DIR, "results_log.csv")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run virtual try-on on multiple image pairs.")
    parser.add_argument("--persons_dir", type=str, default=None, help="Folder of person photos.")
    parser.add_argument("--garments_dir", type=str, default=None, help="Folder of garment photos.")
    parser.add_argument(
        "--pairing",
        type=str,
        choices=["all", "matching"],
        default="all",
        help="'all' = every person x every garment. 'matching' = pair by same index "
        "(1st person with 1st garment, 2nd with 2nd, ...). Default: all.",
    )
    parser.add_argument("--manifest", type=str, default=None, help="CSV file with explicit person,garment,description rows.")
    parser.add_argument(
        "--description",
        type=str,
        default="a garment",
        help="Default garment description used in folder mode (edit per-item in manifest mode instead).",
    )
    return parser.parse_args()


def list_images(folder: str):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not files:
        raise ValueError(f"No images found in: {folder}")
    return files


def build_jobs_from_folders(persons_dir: str, garments_dir: str, pairing: str, description: str):
    persons = list_images(persons_dir)
    garments = list_images(garments_dir)

    jobs = []
    if pairing == "matching":
        n = min(len(persons), len(garments))
        if len(persons) != len(garments):
            print(f"[WARN] persons ({len(persons)}) and garments ({len(garments)}) counts differ; "
                  f"using first {n} matching pairs.")
        for i in range(n):
            jobs.append((persons[i], garments[i], description))
    else:
        for p in persons:
            for g in garments:
                jobs.append((p, g, description))
    return jobs


def build_jobs_from_manifest(manifest_path: str):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    jobs = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append((row["person"], row["garment"], row.get("description", "a garment")))
    if not jobs:
        raise ValueError(f"Manifest file is empty: {manifest_path}")
    return jobs


def main() -> int:
    args = parse_args()

    try:
        if args.manifest:
            jobs = build_jobs_from_manifest(args.manifest)
        elif args.persons_dir and args.garments_dir:
            jobs = build_jobs_from_folders(args.persons_dir, args.garments_dir, args.pairing, args.description)
        else:
            print("[FATAL] Provide either --manifest, or both --persons_dir and --garments_dir.")
            return 1
    except Exception as exc:
        print(f"[FATAL] {exc}")
        return 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Found {len(jobs)} job(s) to run. Results will be saved under: {RESULTS_DIR}")
    print("=" * 70)

    log_rows = []

    for idx, (person_path, garment_path, description) in enumerate(jobs, start=1):
        person_name = os.path.splitext(os.path.basename(person_path))[0]
        garment_name = os.path.splitext(os.path.basename(garment_path))[0]
        out_name = f"{idx:03d}_{person_name}__{garment_name}.jpg"
        out_path = os.path.join(RESULTS_DIR, out_name)

        print(f"[{idx}/{len(jobs)}] {person_path}  +  {garment_path}  ('{description}')")

        status = "OK"
        elapsed = None
        try:
            person_img = cv2.imread(person_path)
            if person_img is None:
                raise ValueError(f"Could not read person image: {person_path}")

            t0 = time.time()
            result = run_tryon(person_image=person_img, garment_image_path=garment_path, garment_description=description)
            elapsed = time.time() - t0

            cv2.imwrite(out_path, result)
            print(f"    -> saved: {out_path}  ({elapsed:.1f}s)")
        except Exception as exc:
            status = f"FAILED: {exc}"
            print(f"    -> [ERROR] {exc}")

        log_rows.append({
            "index": idx,
            "person": person_path,
            "garment": garment_path,
            "description": description,
            "output_file": out_path if status == "OK" else "",
            "status": status,
            "seconds": f"{elapsed:.1f}" if elapsed else "",
            "manual_quality_score_1_to_5": "",  # fill this in by hand after reviewing images
            "notes": "",
        })

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print("=" * 70)
    ok_count = sum(1 for r in log_rows if r["status"] == "OK")
    print(f"Done: {ok_count}/{len(log_rows)} succeeded.")
    print(f"Log written to: {LOG_PATH}")
    print("Open the log in Excel, look at each saved image, and fill in "
          "'manual_quality_score_1_to_5' for your evaluation table.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
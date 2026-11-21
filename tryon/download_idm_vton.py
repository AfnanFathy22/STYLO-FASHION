"""
tryon/download_idm_vton.py
----------------------------
Standalone, resilient downloader for IDM-VTON's weights.

Why this exists separately from setup_models.ps1's inline one-liner:
the default huggingface_hub read timeout is only 10 seconds, which is way
too short for large files (100-250MB+) on a slow/unstable connection --
that's exactly what caused the repeated "Read timed out" errors you saw.

This script:
  - raises the timeout to 120s
  - turns on hf_transfer (a much faster, more resilient Rust-based
    downloader) if installed
  - is SAFE TO RE-RUN as many times as you need -- huggingface_hub resumes
    partially-downloaded files automatically instead of starting over.

Usage:
    python tryon/download_idm_vton.py
"""

import os

# Must be set BEFORE importing huggingface_hub.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

# If hf_transfer is installed (pip install hf_transfer), this switches to a
# much more robust multi-connection downloader that handles flaky networks
# far better than the default. If it's not installed, this line is ignored
# and the normal downloader is used.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_TARGET_DIR = os.path.join(_PROJECT_ROOT, "third_party", "IDM-VTON", "ckpt", "idm-vton")

if __name__ == "__main__":
    print(f"Downloading yisol/IDM-VTON -> {_TARGET_DIR}")
    print("This is resumable -- if it fails partway, just run this script again.")
    while True:
        try:
            snapshot_download(
                repo_id="yisol/IDM-VTON",
                local_dir=_TARGET_DIR,
                max_workers=2,  # fewer parallel connections = more stable on slow links
            )
            print("Done.")
            break
        except Exception as exc:
            print(f"\n[RETRY] Download hiccup: {exc}\nRe-running automatically...\n")

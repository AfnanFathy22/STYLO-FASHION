"""
tryon/download_ootd.py
------------------------
Same idea as tryon/download_idm_vton.py, for OOTDiffusion's checkpoints.
Safe to re-run if it fails partway -- resumes instead of restarting.

Usage:
    python tryon/download_ootd.py
"""

import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_TARGET_DIR = os.path.join(_PROJECT_ROOT, "third_party", "OOTDiffusion", "checkpoints")

if __name__ == "__main__":
    print(f"Downloading levihsu/OOTDiffusion -> {_TARGET_DIR}")
    print("This is resumable -- if it fails partway, just run this script again.")
    while True:
        try:
            snapshot_download(
                repo_id="levihsu/OOTDiffusion",
                local_dir=_TARGET_DIR,
                max_workers=2,
            )
            print("Done.")
            break
        except Exception as exc:
            print(f"\n[RETRY] Download hiccup: {exc}\nRe-running automatically...\n")

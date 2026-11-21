# Integrating virtual try-on into this project

## Quick start (Windows / PowerShell, matches your setup)

```powershell
cd C:\Users\Admin\Desktop\Stage1\Stage1
.\tryon\setup_models.ps1
```

This clones both repos into `third_party\`, installs their Python
requirements into your currently-activated venv, and downloads their
checkpoints from Hugging Face. It takes a while (large downloads) and
needs ~40-50 GB free disk space for both models combined.

`tryon/idm_vton_wrapper.py` and `tryon/ootd_wrapper.py` are now REAL
implementations (not skeletons), adapted from each project's own official
demo/inference script. Read the important caveat below before running them.

## IMPORTANT: verify the official demo works FIRST

Both wrappers are adapted from the official demo scripts as of when this
was written, but these repos change their internals (file names, class
names, config paths) over time, and I can't run/test against your actual
checkout. Before pointing `run_video.py --tryon` at either wrapper:

1. Go into `third_party\IDM-VTON` (or `third_party\OOTDiffusion`) and run
   THAT repo's own example command from its README, standalone, with a
   sample person + garment image.
2. If that works, the environment and weights are good, and any error you
   then get from `tryon/idm_vton_wrapper.py` / `tryon/ootd_wrapper.py`
   is almost certainly an import path or argument mismatch versus what
   your checked-out version expects -- easy to fix by comparing the
   wrapper's imports against the working official script.
3. If the official demo itself doesn't work, fix that first (it's a
   IDM-VTON/OOTDiffusion environment issue, unrelated to this project).

`run_video.py` is already wired to call whichever engine you pick with
`--tryon-model` whenever it detects a significant body change in the
video (see `--tryon` and `--garment` flags):

| `--tryon-model` | Engine        | Wrapper file                     |
|------------------|--------------|-----------------------------------|
| `idm` (default)  | IDM-VTON     | `tryon/idm_vton_wrapper.py`       |
| `ootd`           | OOTDiffusion | `tryon/ootd_wrapper.py`           |

## 1. Hardware reality check (applies to both)

Both are Stable-Diffusion-based models.
- Need an NVIDIA GPU with **at least ~16 GB VRAM** for comfortable
  inference (less is possible with fp16 / offloading, but slower and more
  fragile). OOTDiffusion tends to be somewhat lighter than IDM-VTON but
  still firmly in "needs a real GPU" territory.
- A single try-on image takes roughly **several seconds to ~1 minute**
  depending on GPU and settings -- neither will be real-time on a live
  video stream. That's exactly why `run_video.py` only triggers try-on on
  detected body-change frames instead of every frame.
- No capable local GPU? Consider a rented GPU instance (RunPod, Lambda,
  etc.) or a hosted inference endpoint, and call that from `run_tryon()`
  over HTTP instead of loading weights locally.

## 2. Get the models

**IDM-VTON**
- Repo: https://github.com/yisol/IDM-VTON
- Weights: Hugging Face, `yisol/IDM-VTON`

```bash
git clone https://github.com/yisol/IDM-VTON.git
cd IDM-VTON
conda env create -f environment.yaml   # or pip install -r requirements.txt
# download weights via huggingface-cli / git-lfs as documented in the repo
```

**OOTDiffusion**
- Repo: https://github.com/levihsu/OOTDiffusion
- Weights: Hugging Face, `levihsu/OOTDiffusion`

```bash
git clone https://github.com/levihsu/OOTDiffusion.git
cd OOTDiffusion
pip install -r requirements.txt
# download checkpoints as documented in the repo (separate hd/dc checkpoints)
```

Check each repo's own README for the current, authoritative instructions --
these things drift over time, and version/checkpoint names change.

## 3. Test standalone before wiring into run_video.py

Once `setup_models.ps1` has finished and you've confirmed each repo's own
demo works (see the IMPORTANT section above), test our wrapper directly:

```powershell
python -c "
import cv2
from tryon.idm_vton_wrapper import run_tryon
img = cv2.imread('some_person.jpg')
out = run_tryon(img, 'some_garment.jpg', garment_description='a blue denim jacket')
cv2.imwrite('tryon_result.jpg', out)
"
```
(swap `idm_vton_wrapper` for `ootd_wrapper` to test OOTDiffusion instead --
its `run_tryon()` doesn't take `garment_description`, see its docstring)

Once that produces a sensible `tryon_result.jpg`, run it through the video
pipeline:
```powershell
python run_video.py --video clip.mp4 --garment garment.jpg --tryon --tryon-model idm
# or:
python run_video.py --video clip.mp4 --garment garment.jpg --tryon --tryon-model ootd
```

## 4. Practical tuning

- `--process-every`, `--max-width`, `--max-height` in `run_video.py`
  control the *cheap* pose/seg/mesh models -- keep these reasonably low
  since they're fast.
- `--change-threshold` controls how sensitive body-change detection is.
  Start around `0.03` and adjust: lower = more sensitive (more try-on
  calls, slower overall), higher = only fires on big pose changes.
- Since both engines are slow, consider caching results by a rounded pose
  signature so near-identical poses don't re-trigger generation.
- If you implement both wrappers, it's easy to compare them: run the same
  clip twice with `--tryon-model idm` then `--tryon-model ootd` and diff
  the saved outputs.


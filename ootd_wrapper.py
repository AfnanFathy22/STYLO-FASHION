"""
tryon/ootd_wrapper.py
----------------------
Thin integration layer between this project's Stage 1 pipeline and
OOTDiffusion (https://github.com/levihsu/OOTDiffusion), a diffusion-based
outfitting/virtual try-on model.

STATUS: skeleton only. Follow tryon/README.md to install OOTDiffusion, then
fill in `_load_pipeline()` and `run_tryon()` below.

Same design as tryon/idm_vton_wrapper.py -- kept as a SEPARATE, independent
module on purpose so you can:
  * install/try only one of the two models first,
  * swap between them at runtime with `--tryon-model idm` / `--tryon-model ootd`
    in run_video.py,
  * compare quality/speed side by side later without the two integrations
    interfering with each other.

Design notes specific to OOTDiffusion:
  * It ships two "model types": half-body and full-body (dc / hd checkpoints
    in the original repo). Full-body is the relevant one for this project
    since Stage 1 already does full-body pose/segmentation.
  * Like IDM-VTON, it needs a person image + a garment image, and benefits
    from a person mask/pose as conditioning -- reuse Stage 1's
    PersonSegmenter / PoseEstimator outputs instead of recomputing them.
  * Also SDXL/diffusion based -- same "don't call this every frame" caveat
    as IDM-VTON applies here.
"""

import numpy as np

_pipeline = None  # lazy-loaded singleton


def _load_pipeline():
    """
    Load and cache the OOTDiffusion pipeline.

    Fill this in once you've followed tryon/README.md, e.g. (pseudocode,
    check OOTDiffusion's own repo for exact class names/paths -- they
    changed a few times across versions):

        import sys
        sys.path.append("/path/to/cloned/OOTDiffusion")  # if not pip-installed
        from ootd.inference_ootd_hd import OOTDiffusionHD  # or _dc for full body

        pipeline = OOTDiffusionHD(gpu_id=0)
        return pipeline
    """
    global _pipeline
    if _pipeline is None:
        raise NotImplementedError(
            "OOTDiffusion pipeline loading isn't implemented yet. "
            "Follow tryon/README.md to install the model, then implement "
            "_load_pipeline() in tryon/ootd_wrapper.py."
        )
    return _pipeline


def run_tryon(person_image: np.ndarray, garment_image_path: str) -> np.ndarray:
    """
    Run virtual try-on with OOTDiffusion: put the garment at
    garment_image_path onto the person in person_image (a BGR OpenCV frame,
    as produced by run_video.py / run_image.py).

    Returns a BGR numpy array with the person now wearing the garment,
    or raises on failure. Same calling convention as
    tryon/idm_vton_wrapper.run_tryon() on purpose, so run_video.py can call
    either one interchangeably.

    TODO once OOTDiffusion is installed:
      1. Convert person_image (BGR numpy) -> PIL RGB image.
      2. Load the garment image from garment_image_path -> PIL RGB image.
      3. (Recommended) Reuse this project's existing PersonSegmenter /
         PoseEstimator outputs instead of letting OOTDiffusion recompute
         its own human parsing from scratch.
      4. Call the pipeline, e.g.:
             images = pipeline(
                 model_type="hd",              # or "dc" for full body
                 category="upper_body",        # or "lower_body" / "dresses"
                 image_garm=garment_pil,
                 image_vton=person_pil,
                 mask=person_mask,              # from PersonSegmenter
                 image_ori=person_pil,
                 num_samples=1,
                 num_steps=20,
             )
             result = images[0]
      5. Convert the PIL result back to BGR numpy and return it.
    """
    pipeline = _load_pipeline()  # will raise until implemented
    raise NotImplementedError("run_tryon() body not implemented yet.")

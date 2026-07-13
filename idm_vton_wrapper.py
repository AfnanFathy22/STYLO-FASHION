"""
tryon/idm_vton_wrapper.py
--------------------------
Thin integration layer between this project's Stage 1 pipeline and IDM-VTON
(https://github.com/yisol/IDM-VTON), a diffusion-based virtual try-on model.

STATUS: skeleton only. Follow tryon/README.md to install IDM-VTON, then
fill in `_load_pipeline()` and `run_tryon()` below.

Design notes:
  * IDM-VTON is a Stable-Diffusion-XL-based inpainting/try-on model. A single
    inference pass typically takes several seconds even on a good GPU, and
    needs a person image, a garment image, and (usually) a person-pose /
    parsing map as extra conditioning -- this is exactly what Stage 1
    (PoseEstimator + PersonSegmenter) already produces, so plug their
    outputs in here instead of recomputing anything.
  * Because of the per-call cost, run_video.py only calls run_tryon() on
    frames flagged as a "body change", not on every frame.
  * The heavy pipeline object should be created ONCE (lazy singleton) and
    reused across calls -- reloading SDXL weights per frame would be far
    too slow.
"""

import numpy as np

_pipeline = None  # lazy-loaded singleton


def _load_pipeline():
    """
    Load and cache the IDM-VTON diffusers pipeline.

    Fill this in once you've followed tryon/README.md, e.g. (pseudocode,
    check IDM-VTON's own repo for the exact class names/paths, they change
    between versions):

        from diffusers import DDPMScheduler, AutoencoderKL
        from idm_vton.pipeline import IDMVTONPipeline  # from the cloned repo

        pipeline = IDMVTONPipeline.from_pretrained(
            "yisol/IDM-VTON",
            torch_dtype=torch.float16,
        ).to("cuda")
        return pipeline
    """
    global _pipeline
    if _pipeline is None:
        raise NotImplementedError(
            "IDM-VTON pipeline loading isn't implemented yet. "
            "Follow tryon/README.md to install the model, then implement "
            "_load_pipeline() in tryon/idm_vton_wrapper.py."
        )
    return _pipeline


def run_tryon(person_image: np.ndarray, garment_image_path: str) -> np.ndarray:
    """
    Run virtual try-on: put the garment at garment_image_path onto the
    person in person_image (a BGR OpenCV frame, as produced by run_video.py
    / run_image.py).

    Returns a BGR numpy array (same convention as the input) with the
    person now wearing the garment, or raises on failure.

    TODO once IDM-VTON is installed:
      1. Convert person_image (BGR numpy) -> PIL RGB image.
      2. Load the garment image from garment_image_path -> PIL RGB image.
      3. (Recommended) Reuse this project's existing PersonSegmenter /
         PoseEstimator outputs for the person mask & pose, since IDM-VTON's
         own preprocessing (human parsing + OpenPose) does the same job --
         no need to run it twice if you already have it in Stage 1.
      4. Call the pipeline, e.g.:
             result = pipeline(
                 person_image=person_pil,
                 garment_image=garment_pil,
                 mask=person_mask,          # from PersonSegmenter
                 pose=person_pose,          # from PoseEstimator
                 num_inference_steps=30,
             ).images[0]
      5. Convert the PIL result back to BGR numpy and return it.
    """
    pipeline = _load_pipeline()  # will raise until implemented
    raise NotImplementedError("run_tryon() body not implemented yet.")

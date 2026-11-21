"""
tryon/ootd_wrapper.py
----------------------
Working integration with OOTDiffusion (https://github.com/levihsu/OOTDiffusion).

BEFORE USING THIS FILE:
  1. Run tryon/setup_models.ps1 (or clone/download manually per
     tryon/README.md) so third_party/OOTDiffusion exists with its
     checkpoints.
  2. Get OOTDiffusion's OWN run/inference.py working standalone first
     (via its README's example command) before using this wrapper --
     this file mirrors that script's logic, so if the original doesn't
     run, this wrapper won't either.

Like the IDM-VTON wrapper, this lets OOTDiffusion run its OWN bundled
human-parsing + OpenPose preprocessing on the frame, rather than trying to
reuse our Stage 1 (MediaPipe-based) pose/segmentation output -- the two
projects use different label formats and OOTDiffusion's own preprocessing
is what its model was actually trained against.

NOTE: OOTDiffusion's public code has had both a "half body" (hd) and
"dress code / full body" (dc) checkpoint set at different times, with
slightly different class names. If an import here fails, check
third_party/OOTDiffusion/run/inference.py for the current class names and
adjust the import below to match.
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_OOTD_DIR = os.path.join(_PROJECT_ROOT, "third_party", "OOTDiffusion")

_state = {}  # lazily filled by _load_pipeline()


def _load_pipeline(model_type: str = "hd"):
    """
    model_type: "hd" (half/upper body, higher quality) or "dc" (dress code /
    full body, needed for lower_body / dresses categories).
    """
    cache_key = f"pipe_{model_type}"
    if cache_key in _state:
        return _state

    if not os.path.isdir(_OOTD_DIR):
        raise RuntimeError(
            f"OOTDiffusion repo not found at {_OOTD_DIR}. Run tryon/setup_models.ps1 first, "
            "or clone it manually per tryon/README.md."
        )
    # OOTDiffusion's own code imports things relative to its own root and a
    # couple of subfolders, so both need to be on sys.path.
    for p in (_OOTD_DIR, os.path.join(_OOTD_DIR, "run")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch

    if model_type == "hd":
        from ootd.inference_ootd_hd import OOTDiffusionHD as OOTDiffusionModel
    else:
        from ootd.inference_ootd_dc import OOTDiffusionDC as OOTDiffusionModel

    from preprocess.openpose.run_openpose import OpenPose
    from preprocess.humanparsing.run_parsing import Parsing

    gpu_id = 0 if torch.cuda.is_available() else -1
    print(f"[OOTDiffusion] Loading '{model_type}' pipeline (gpu_id={gpu_id})...")

    pipeline = OOTDiffusionModel(gpu_id)
    openpose_model = OpenPose(gpu_id)
    parsing_model = Parsing(gpu_id)

    _state.clear()  # only keep one model_type loaded at a time (VRAM)
    _state.update(
        pipe=pipeline,
        openpose_model=openpose_model,
        parsing_model=parsing_model,
        model_type=model_type,
        torch=torch,
    )
    print("[OOTDiffusion] Pipeline ready.")
    return _state


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def run_tryon(
    person_image: np.ndarray,
    garment_image_path: str,
    category: str = "upper_body",
    model_type: str = "hd",
    num_samples: int = 1,
    num_steps: int = 20,
    scale: float = 2.0,
    seed: int = -1,
) -> np.ndarray:
    """
    Run OOTDiffusion on a single frame.

    person_image: BGR OpenCV frame (what run_video.py passes in).
    garment_image_path: path to a garment photo (plain background works best).
    category: "upper_body" | "lower_body" | "dresses" (OOTDiffusion convention).
        NOTE: "lower_body" and "dresses" require model_type="dc".
    model_type: "hd" (upper_body only, better quality) or "dc" (all categories).

    Returns a BGR numpy array, same convention as the input.
    """
    if category != "upper_body" and model_type != "dc":
        print(
            f"[OOTDiffusion] category='{category}' needs model_type='dc' -- switching automatically."
        )
        model_type = "dc"

    state = _load_pipeline(model_type)
    pipeline = state["pipe"]
    openpose_model = state["openpose_model"]
    parsing_model = state["parsing_model"]

    category_map = {"upper_body": 0, "lower_body": 1, "dresses": 2}
    if category not in category_map:
        raise ValueError(f"category must be one of {list(category_map)}, got '{category}'")

    garm_img = Image.open(garment_image_path).convert("RGB").resize((768, 1024))
    person_img = _bgr_to_pil(person_image).convert("RGB").resize((384, 512))

    keypoints = openpose_model(person_img)
    model_parse, _ = parsing_model(person_img)

    # OOTDiffusion's own repo bundles a helper (get_mask_location) used the
    # same way as IDM-VTON's -- see third_party/OOTDiffusion/run/utils_ootd.py
    # (name may differ slightly by version; check that file if this import fails).
    from utils_ootd import get_mask_location

    mask, mask_gray = get_mask_location(
        model_type, category_map_name(category), model_parse, keypoints
    )
    mask = mask.resize((768, 1024))
    mask_gray = mask_gray.resize((768, 1024))

    person_img_full = _bgr_to_pil(person_image).convert("RGB").resize((768, 1024))
    masked_vton_img = Image.composite(mask_gray, person_img_full, mask)

    images = pipeline(
        model_type=model_type,
        category=category_map[category],
        image_garm=garm_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=person_img_full,
        num_samples=num_samples,
        num_steps=num_steps,
        image_scale=scale,
        seed=seed,
    )

    return _pil_to_bgr(images[0])


def category_map_name(category: str) -> str:
    """OOTDiffusion's get_mask_location expects the category as a string
    name in some versions of the repo -- kept as a separate small helper so
    it's easy to swap to the integer form above if your checked-out version
    expects that instead."""
    return category

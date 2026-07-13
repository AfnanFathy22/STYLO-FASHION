"""
tryon/idm_vton_wrapper.py
--------------------------
Working integration with IDM-VTON (https://github.com/yisol/IDM-VTON).

BEFORE USING THIS FILE:
  1. Run tryon/setup_models.ps1 (or clone/download manually per
     tryon/README.md) so third_party/IDM-VTON exists with its checkpoints.
  2. Get IDM-VTON's OWN demo (app.py / gradio) running standalone first.
     This file's code closely follows that demo's logic, adapted to accept
     an OpenCV BGR frame instead of a Gradio UI upload -- if the official
     demo doesn't run for you, this wrapper won't either, and the fix
     belongs in your IDM-VTON install, not here.

WHY IT DOESN'T REUSE OUR OWN PoseEstimator/PersonSegmenter:
  IDM-VTON's pipeline expects VERY specific conditioning: a DensePose
  segmentation image (not a generic pose skeleton) and a human-parsing mask
  in its own label format. These come from modules bundled inside the
  IDM-VTON repo itself (preprocess/humanparsing, preprocess/openpose, plus
  detectron2 DensePose). Re-deriving that exact format from our own
  MediaPipe-based Stage 1 output would be fragile and likely subtly wrong,
  so this wrapper lets IDM-VTON run its own preprocessing on the frame --
  it's slower, but correct. Our Stage 1 pipeline still does its job earlier
  in run_video.py (driving WHEN a try-on call fires via body-change
  detection); it just isn't reused as VTON conditioning here.

NOTE: IDM-VTON's public code has moved around a few times (file/class
names, exact config paths). If an import here fails, open
third_party/IDM-VTON and check the current path for that class in their
app.py / inference.py, then adjust the import below to match.
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_IDM_DIR = os.path.join(_PROJECT_ROOT, "third_party", "IDM-VTON")
_CKPT_DIR = os.path.join(_IDM_DIR, "ckpt", "idm-vton")

_state = {}  # lazily filled by _load_pipeline()


def _load_pipeline():
    if "pipe" in _state:
        return _state

    if not os.path.isdir(_IDM_DIR):
        raise RuntimeError(
            f"IDM-VTON repo not found at {_IDM_DIR}. Run tryon/setup_models.ps1 first, "
            "or clone it manually per tryon/README.md."
        )
    if _IDM_DIR not in sys.path:
        sys.path.insert(0, _IDM_DIR)

    import torch
    from transformers import (
        AutoTokenizer,
        CLIPImageProcessor,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    )
    from diffusers import DDPMScheduler, AutoencoderKL

    # These come from inside third_party/IDM-VTON -- see the note above if
    # any of these paths have changed in the upstream repo.
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    base_path = _CKPT_DIR if os.path.isdir(_CKPT_DIR) else "yisol/IDM-VTON"

    print(f"[IDM-VTON] Loading weights from: {base_path} (device={device})")

    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=dtype)
    unet.requires_grad_(False)

    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path, subfolder="unet_encoder", torch_dtype=dtype
    )
    unet_encoder.requires_grad_(False)

    tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
    text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path, subfolder="text_encoder_2", torch_dtype=dtype
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path, subfolder="image_encoder", torch_dtype=dtype
    )
    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=dtype)

    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    pipe.unet_encoder = unet_encoder
    pipe.to(device)

    parsing_model = Parsing(0 if device == "cuda" else -1)
    openpose_model = OpenPose(0 if device == "cuda" else -1)

    _state.update(
        pipe=pipe,
        parsing_model=parsing_model,
        openpose_model=openpose_model,
        device=device,
        dtype=dtype,
        torch=torch,
    )
    print("[IDM-VTON] Pipeline ready.")
    return _state


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def run_tryon(
    person_image: np.ndarray,
    garment_image_path: str,
    garment_description: str = "a garment",
    category: str = "upper_body",
    num_inference_steps: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """
    Run IDM-VTON on a single frame.

    person_image: BGR OpenCV frame (what run_video.py passes in).
    garment_image_path: path to a garment photo (plain background works best).
    garment_description: short text like "a red knit sweater" -- IDM-VTON
        uses this as part of its text conditioning; a decent description
        noticeably improves quality.
    category: "upper_body" | "lower_body" | "dresses" (IDM-VTON convention).

    Returns a BGR numpy array, same convention as the input.
    """
    state = _load_pipeline()
    pipe = state["pipe"]
    parsing_model = state["parsing_model"]
    openpose_model = state["openpose_model"]
    device = state["device"]
    dtype = state["dtype"]
    torch = state["torch"]

    from utils_mask import get_mask_location  # from third_party/IDM-VTON

    human_img = _bgr_to_pil(person_image).convert("RGB").resize((768, 1024))
    garm_img = Image.open(garment_image_path).convert("RGB").resize((768, 1024))

    keypoints = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    mask, _mask_gray = get_mask_location("hd", category, model_parse, keypoints)
    mask = mask.resize((768, 1024))

    # DensePose-based pose image. IDM-VTON's own repo provides `apply_net.py`
    # at its root for this -- see third_party/IDM-VTON/apply_net.py for the
    # exact CLI it expects if this import/call needs adjusting on your setup.
    import apply_net

    densepose_args = apply_net.create_argument_parser().parse_args((
        "show",
        os.path.join(_IDM_DIR, "configs", "densepose_rcnn_R_50_FPN_s1x.yaml"),
        os.path.join(_IDM_DIR, "ckpt", "densepose", "model_final_162be9.pkl"),
        "dp_segm", "-v", "--opts", "MODEL.DEVICE", device,
    ))
    human_img_np = np.array(human_img)[:, :, ::-1]  # RGB -> BGR for densepose helper
    pose_img_np = densepose_args.func(densepose_args, human_img_np)
    pose_img = Image.fromarray(pose_img_np[:, :, ::-1]).resize((768, 1024))

    generator = torch.Generator(device).manual_seed(seed)

    with torch.no_grad():
        prompt = f"model is wearing {garment_description}"
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        )

        cloth_prompt = f"a photo of {garment_description}"
        (prompt_embeds_c, _neg_c, _pooled_c, _neg_pooled_c) = pipe.encode_prompt(
            cloth_prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        )

        garm_tensor = pipe.image_processor.preprocess(garm_img).to(device, dtype)
        pose_tensor = (
            torch.from_numpy(np.array(pose_img)).permute(2, 0, 1).unsqueeze(0).to(device, dtype) / 255.0
        )

        result = pipe(
            prompt_embeds=prompt_embeds.to(device, dtype),
            negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
            pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
            num_inference_steps=num_inference_steps,
            generator=generator,
            strength=1.0,
            pose_img=pose_tensor,
            text_embeds_cloth=prompt_embeds_c.to(device, dtype),
            cloth=garm_tensor,
            mask_image=mask,
            image=human_img,
            height=1024,
            width=768,
            ip_adapter_image=garm_img,
            guidance_scale=2.0,
        ).images[0]

    return _pil_to_bgr(result)

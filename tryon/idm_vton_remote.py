"""
tryon/idm_vton_remote.py
------------------------
Remote alternative to idm_vton_wrapper.py. Instead of running IDM-VTON's
diffusion pipeline locally (needs a GPU + detectron2, both painful on a
CPU-only Windows machine), this calls the OFFICIAL IDM-VTON Hugging Face
Space over the network and gets the result image back.

No local model download, no detectron2, no GPU required on your machine --
just an internet connection.

Same `run_tryon(person_image, garment_image_path, ...)` interface as
idm_vton_wrapper.py, so run_video.py can call either one interchangeably.

Requirements:
    pip install gradio_client

Notes on the underlying Space (https://huggingface.co/spaces/yisol/IDM-VTON):
  - It's a shared, free, ZeroGPU-powered Space. It can queue behind other
    users' requests, and Hugging Face rate-limits anonymous usage.
  - If you hit rate limits or long queues, get a free Hugging Face account
    and token (https://huggingface.co/settings/tokens) and set it as an
    environment variable before running your script:
        setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxxxxx"      (Windows, permanent)
        $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"      (current PowerShell session)
    A token gives you your own request queue instead of the fully public one.
"""

import os
import tempfile

import cv2
import numpy as np

_SPACE_ID = "yisol/IDM-VTON"  # the official IDM-VTON demo Space
_client = None  # lazily created so importing this module never touches the network


def _get_client():
    """Create (once) and cache the gradio_client.Client connected to the Space."""
    global _client
    if _client is not None:
        return _client

    try:
        from gradio_client import Client
    except ImportError as exc:
        raise ImportError(
            "gradio_client is required for the remote try-on wrapper. "
            "Install it with: pip install gradio_client"
        ) from exc

    hf_token = os.environ.get("HF_TOKEN")  # optional; None works for public/anonymous use

    try:
        print(f"[IDM-VTON-Remote] Connecting to Hugging Face Space '{_SPACE_ID}'...")
        _client = Client(_SPACE_ID, hf_token=hf_token)
        print("[IDM-VTON-Remote] Connected.")
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to the Hugging Face Space '{_SPACE_ID}'. "
            f"Check your internet connection and that the Space is currently running "
            f"(visit https://huggingface.co/spaces/{_SPACE_ID} in a browser to confirm). "
            f"Original error: {exc}"
        ) from exc

    return _client


def run_tryon(
    person_image: np.ndarray,
    garment_image_path: str,
    garment_description: str = "",
    use_auto_mask: bool = True,
    use_auto_crop: bool = False,
    num_inference_steps: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """
    Run virtual try-on via the remote IDM-VTON Hugging Face Space.

    Args:
        person_image: BGR OpenCV frame/image (the person photo).
        garment_image_path: path to a garment photo (plain background works best).
        garment_description: short text description of the garment, e.g.
            "a black dress" -- improves quality, but can be left empty.
        use_auto_mask: let the Space auto-generate the inpainting mask
            (recommended; matches the demo's default "Yes").
        use_auto_crop: let the Space auto-crop/resize the person image
            (useful for full-body photos where the person is small in frame).
        num_inference_steps: diffusion denoising steps (Space default: 30).
        seed: random seed for reproducibility.

    Returns:
        BGR numpy array (same convention as the input) with the generated
        try-on result.
    """
    if person_image is None or person_image.size == 0:
        raise ValueError("person_image is empty/None.")
    if not os.path.exists(garment_image_path):
        raise FileNotFoundError(f"Garment image not found at '{garment_image_path}'.")

    from gradio_client import handle_file

    client = _get_client()

    # gradio_client's handle_file() needs a path on disk, so write the
    # in-memory person_image out to a temporary PNG first.
    tmp_person_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_person_path = tmp.name
        if not cv2.imwrite(tmp_person_path, person_image):
            raise RuntimeError("Failed to write the person image to a temporary file.")

        print("[IDM-VTON-Remote] Sending request (this can take 10-60+ seconds, "
              "depending on the Space's current queue)...")

        result = client.predict(
            dict={
                "background": handle_file(tmp_person_path),
                "layers": [],
                "composite": None,
            },
            garm_img=handle_file(garment_image_path),
            garment_des=garment_description,
            is_checked=use_auto_mask,
            is_checked_crop=use_auto_crop,
            denoise_steps=num_inference_steps,
            seed=seed,
            api_name="/tryon",
        )

    except Exception as exc:
        raise RuntimeError(
            f"IDM-VTON remote try-on request failed: {exc}\n"
            "Common causes: the Space is asleep/overloaded (try again in a minute), "
            "no internet connection, or the Space's API changed. You can check the "
            f"Space directly at https://huggingface.co/spaces/{_SPACE_ID} to see if "
            "it's currently responsive."
        ) from exc
    finally:
        if tmp_person_path and os.path.exists(tmp_person_path):
            os.remove(tmp_person_path)

    # result is a tuple: (output_image_path, masked_image_path)
    output_image_path = result[0]
    output_bgr = cv2.imread(output_image_path)
    if output_bgr is None:
        raise RuntimeError(
            f"Received a result from the Space but could not read the output image "
            f"at '{output_image_path}'."
        )

    print("[IDM-VTON-Remote] Done.")
    return output_bgr
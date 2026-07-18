"""
inpaint_kontext.py — FLUX Kontext / Fill backends via fal.

- masked (default): fal-ai/flux-pro/v1/fill
- instruction: fal-ai/flux-kontext/dev (no strict mask)
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Literal, Union

from PIL import Image

from backend.inpaint import (
    KONTEXT_EDIT_MODEL,
    KONTEXT_FILL_MODEL,
    InpaintError,
    InpaintResult,
    composite_masked,
    download_image,
    extract_image_url,
    hash_image_bytes,
    load_image,
    new_request_id,
    normalize_mask,
    resolve_fal_key,
    save_inpaint_output,
    save_mask_sidecar,
    subscribe_fal,
    upload_pil,
)
from backend.segment import MaskInstance

KontextMode = Literal["masked", "instruction"]


def _download_resized(out_url: str, timeout: int, size: tuple[int, int]) -> Image.Image:
    result_img = download_image(out_url, timeout=timeout)
    if result_img.size != size:
        result_img = result_img.resize(size, Image.LANCZOS)
    return result_img


def inpaint(
    image: Union[str, Path, bytes, Image.Image],
    mask: Union[str, Path, Image.Image, MaskInstance] | None,
    prompt: str,
    *,
    mode: KontextMode = "masked",
    negative_prompt: str | None = None,
    seed: int | None = None,
    reference_image: Union[str, Path, bytes, Image.Image] | None = None,
    parent_step_id: str | None = None,
    api_key: str | None = None,
    timeout: int = 180,
) -> InpaintResult:
    """
    FLUX Kontext path.

    mode="masked" requires a mask and uses flux-pro/v1/fill.
    mode="instruction" ignores mask for the API call (edit via prompt only).
    """
    del negative_prompt  # fill/edit endpoints do not take SD-style negatives
    resolve_fal_key(api_key)
    base = load_image(image)

    request_id = new_request_id()
    mask_path = None
    mask_img: Image.Image | None = None
    model_id: str

    if mode == "instruction":
        model_id = KONTEXT_EDIT_MODEL
        image_url = upload_pil(base, suffix=".png")
        arguments: dict = {
            "prompt": prompt,
            "image_url": image_url,
        }
        if seed is not None:
            arguments["seed"] = seed
        if reference_image is not None:
            arguments["prompt"] = (
                f"{prompt} (preserve identity/style from the reference when possible)"
            )
    else:
        if mask is None:
            raise InpaintError("flux_kontext masked mode requires a mask")
        model_id = KONTEXT_FILL_MODEL
        mask_img = normalize_mask(mask, base.size)
        mask_path = str(save_mask_sidecar(mask_img, request_id))
        image_url = upload_pil(base, suffix=".png")
        mask_url = upload_pil(mask_img.convert("RGB"), suffix=".png")
        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "mask_url": mask_url,
            "num_images": 1,
            "output_format": "png",
            "safety_tolerance": "5",
        }
        if seed is not None:
            arguments["seed"] = seed

    data, fal_req_id = subscribe_fal(model_id, arguments, timeout=timeout)
    out_url = extract_image_url(data)
    result_img = _download_resized(out_url, timeout, base.size)

    if mode == "masked" and mask_img is not None:
        result_img = composite_masked(base, result_img, mask_img)

    out_path = save_inpaint_output(result_img, request_id)

    buf = io.BytesIO()
    base.save(buf, format="PNG")
    image_hash = hash_image_bytes(buf.getvalue())

    return InpaintResult(
        request_id=request_id,
        image_hash=image_hash,
        prompt=prompt,
        negative_prompt=None,
        seed=data.get("seed", seed),
        backend="flux_kontext",
        model_version=model_id,
        timestamp=time.time(),
        output_path=str(out_path),
        mask_path=mask_path,
        parent_step_id=parent_step_id,
        fal_request_id=fal_req_id,
        raw_response={
            "images": data.get("images"),
            "seed": data.get("seed"),
            "mode": mode,
        },
    )

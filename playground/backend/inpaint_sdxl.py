"""
inpaint_sdxl.py — fal-ai/fast-sdxl/inpainting backend.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

from PIL import Image

from backend.inpaint import (
    SDXL_MODEL,
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


def inpaint(
    image: Union[str, Path, bytes, Image.Image],
    mask: Union[str, Path, Image.Image, MaskInstance],
    prompt: str,
    *,
    negative_prompt: str | None = None,
    seed: int | None = None,
    strength: float = 0.95,
    parent_step_id: str | None = None,
    api_key: str | None = None,
    timeout: int = 180,
) -> InpaintResult:
    resolve_fal_key(api_key)
    base = load_image(image)
    mask_img = normalize_mask(mask, base.size)

    image_url = upload_pil(base, suffix=".png")
    mask_url = upload_pil(mask_img.convert("RGB"), suffix=".png")

    arguments: dict = {
        "prompt": prompt,
        "image_url": image_url,
        "mask_url": mask_url,
        "strength": strength,
        "num_images": 1,
        "enable_safety_checker": False,
        "format": "png",
        "image_size": {"width": base.width, "height": base.height},
    }
    if negative_prompt:
        arguments["negative_prompt"] = negative_prompt
    if seed is not None:
        arguments["seed"] = seed

    data, fal_req_id = subscribe_fal(SDXL_MODEL, arguments, timeout=timeout)
    out_url = extract_image_url(data)
    result_img = download_image(out_url, timeout=timeout)

    # If fal resized, snap back to original size for quality-gate comparisons.
    if result_img.size != base.size:
        result_img = result_img.resize(base.size, Image.LANCZOS)

    # Surgical composite: never let the model rewrite outside the mask.
    result_img = composite_masked(base, result_img, mask_img)

    request_id = new_request_id()
    out_path = save_inpaint_output(result_img, request_id)
    mask_path = save_mask_sidecar(mask_img, request_id)

    import io

    buf = io.BytesIO()
    base.save(buf, format="PNG")
    image_hash = hash_image_bytes(buf.getvalue())

    return InpaintResult(
        request_id=request_id,
        image_hash=image_hash,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=data.get("seed", seed),
        backend="sdxl",
        model_version=SDXL_MODEL,
        timestamp=time.time(),
        output_path=str(out_path),
        mask_path=str(mask_path),
        parent_step_id=parent_step_id,
        fal_request_id=fal_req_id,
        raw_response={"images": data.get("images"), "seed": data.get("seed")},
    )

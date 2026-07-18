"""
quality_gate.py — A2 edit-quality score card for masked inpainting.

Signals:
  - outside_mask_fidelity  (local): preserve pixels outside the mask
  - inside_mask_changed    (local): edit actually changed the masked region
  - prompt_adherence       (fal Moondream2): cropped region matches the prompt
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from backend.inpaint import (
    MOONDREAM_MODEL,
    QualityScoreCard,
    load_image,
    resolve_fal_key,
    subscribe_fal,
    upload_pil,
)
from backend.mask_utils import load_mask, mask_bbox

# Defaults from the plan
_OUTSIDE_MAE_PASS = 2.0 / 255.0  # near pixel-identical
_INSIDE_MAE_MIN = 5.0 / 255.0  # reject near no-ops
_PROMPT_PASS = 0.6


def _to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _mae(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    """Mean absolute error over region (boolean mask), scaled 0..1 per channel avg."""
    if not np.any(region):
        return 0.0
    diff = np.abs(a[region] - b[region]) / 255.0
    return float(diff.mean())


def _fidelity_from_mae(mae: float, pass_threshold: float = _OUTSIDE_MAE_PASS) -> float:
    """Map outside MAE to 0..1 where 1 = perfect. Soft falloff past threshold."""
    if mae <= 0:
        return 1.0
    # 1 at 0, ~0.5 at pass_threshold*2, approaches 0
    return float(np.exp(-mae / max(pass_threshold, 1e-6)))


def _change_from_mae(mae: float, min_threshold: float = _INSIDE_MAE_MIN) -> float:
    """Map inside MAE to 0..1 where higher = more change. Saturates around 0.2 MAE."""
    return float(min(1.0, mae / 0.2))


def _parse_vlm_match(text: str) -> tuple[float, str]:
    """Extract match score 0..1 from VLM output; soft-fail to 0 on parse errors."""
    if not text:
        return 0.0, "empty VLM response"
    # Try JSON object first
    try:
        # find first {...}
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            match = float(obj.get("match", obj.get("score", 0)))
            reason = str(obj.get("reason", ""))
            return max(0.0, min(1.0, match)), reason or text[:200]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # Fallback: look for a float 0-1 or 0-10
    m = re.search(r"\b(0?\.\d+|1\.0|1|0)\b", text)
    if m:
        val = float(m.group(1))
        if val > 1.0:
            val = val / 10.0
        return max(0.0, min(1.0, val)), text[:200]
    return 0.0, f"unparseable VLM output: {text[:200]}"


def _crop_masked_region(
    image: Image.Image, mask: Image.Image, padding: int = 8
) -> Image.Image:
    x0, y0, x1, y1 = mask_bbox(mask, padding=padding)
    return image.crop((x0, y0, x1, y1))


def _prompt_adherence_vlm(
    result_image: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    api_key: str | None = None,
) -> tuple[float, str]:
    resolve_fal_key(api_key)
    crop = _crop_masked_region(result_image, mask)
    url = upload_pil(crop, suffix=".png")
    rubric = (
        "You are scoring an image edit. The crop shows ONLY the edited region. "
        f'The intended edit prompt was: "{prompt}". '
        "Reply with ONLY a JSON object like "
        '{"match": 0.0, "reason": "short reason"} where match is from 0 to 1 '
        "(1 = the crop clearly matches the prompt, 0 = completely unrelated)."
    )
    data, _ = subscribe_fal(
        MOONDREAM_MODEL,
        {"image_url": url, "prompt": rubric},
    )
    # moondream returns {"output": "..."} typically
    text = data.get("output") or data.get("text") or data.get("answer") or str(data)
    return _parse_vlm_match(str(text))


def evaluate_inpaint(
    original: Union[str, Path, bytes, Image.Image],
    result: Union[str, Path, bytes, Image.Image],
    mask: Union[str, Path, Image.Image],
    prompt: str,
    *,
    run_vlm: bool = True,
    api_key: str | None = None,
    outside_mae_pass: float = _OUTSIDE_MAE_PASS,
    inside_mae_min: float = _INSIDE_MAE_MIN,
    prompt_pass: float = _PROMPT_PASS,
) -> QualityScoreCard:
    """
    Score a masked inpaint result.

    Local geometric checks always run. If outside fidelity fails hard,
    skip VLM (caller can fall back immediately).
    """
    orig = load_image(original)
    res = load_image(result)
    if res.size != orig.size:
        res = res.resize(orig.size, Image.LANCZOS)

    m = load_mask(mask)
    if m.size != orig.size:
        m = m.resize(orig.size, Image.NEAREST)

    a = _to_rgb_array(orig)
    b = _to_rgb_array(res)
    mask_arr = np.asarray(m, dtype=np.float32)
    inside = mask_arr > 127
    outside = ~inside

    outside_mae = _mae(a, b, outside)
    inside_mae = _mae(a, b, inside)

    outside_fidelity = _fidelity_from_mae(outside_mae, outside_mae_pass)
    inside_changed = _change_from_mae(inside_mae, inside_mae_min)

    reasons: list[str] = []
    if outside_mae > outside_mae_pass:
        reasons.append(
            f"outside_mask_fidelity fail: MAE={outside_mae:.4f} > {outside_mae_pass:.4f}"
        )
    if inside_mae < inside_mae_min:
        reasons.append(
            f"inside_mask_changed fail: MAE={inside_mae:.4f} < {inside_mae_min:.4f} (no-op?)"
        )

    prompt_score = 1.0
    raw_vlm = None
    skip_vlm = outside_mae > outside_mae_pass * 3  # hard bleed → skip VLM

    if run_vlm and not skip_vlm:
        try:
            prompt_score, raw_vlm = _prompt_adherence_vlm(
                res, m, prompt, api_key=api_key
            )
            if prompt_score < prompt_pass:
                reasons.append(
                    f"prompt_adherence fail: {prompt_score:.2f} < {prompt_pass}"
                )
        except Exception as e:
            prompt_score = 0.0
            raw_vlm = f"VLM error: {e}"
            reasons.append(f"prompt_adherence fail: VLM error ({e})")
    elif skip_vlm:
        prompt_score = 0.0
        raw_vlm = "skipped VLM due to severe outside-mask bleed"
        reasons.append("prompt_adherence skipped (severe outside bleed)")
    else:
        # gate local-only: treat prompt as pass so it doesn't block
        prompt_score = 1.0

    passed = (
        outside_mae <= outside_mae_pass
        and inside_mae >= inside_mae_min
        and prompt_score >= prompt_pass
    )

    return QualityScoreCard(
        outside_mask_fidelity=outside_fidelity,
        inside_mask_changed=inside_changed,
        prompt_adherence=prompt_score,
        passed=passed,
        reasons=reasons,
        raw_vlm=raw_vlm,
    )

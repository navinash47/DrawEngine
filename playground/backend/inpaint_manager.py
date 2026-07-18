"""
inpaint_manager.py — Routes masked inpainting across backends + quality gate.

Primary: SDXL (inpaint_sdxl.py)
Fallback: FLUX Kontext fill (inpaint_kontext.py)
Stub: AnimeAdapter (inpaint_animeadapter.py)

auto:
  - surface edits: SDXL → quality gate → on fail/error → Kontext → re-score → best-of
  - structural edits: skip SDXL, expand mask 2x, go straight to Kontext
Writes provenance exactly once per successful public inpaint() call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

from PIL import Image

from backend import inpaint_animeadapter, inpaint_kontext, inpaint_sdxl
from backend.edit_classifier import classify_edit_type
from backend.inpaint import (
    InpaintError,
    InpaintResult,
    append_provenance,
    load_image,
    normalize_mask,
)
from backend.mask_utils import expand_mask, mask_expansion_clipped
from backend.quality_gate import evaluate_inpaint
from backend.segment import MaskInstance

BackendName = Literal["auto", "sdxl", "flux_kontext", "animeadapter"]
VALID_BACKENDS = ("auto", "sdxl", "flux_kontext", "animeadapter")
KontextMode = Literal["masked", "instruction"]


def _attach_gate(
    result: InpaintResult,
    original: Image.Image,
    mask_img: Image.Image,
    prompt: str,
    *,
    run_quality_gate: bool,
    api_key: str | None,
) -> InpaintResult:
    if not run_quality_gate:
        result.score_card = None
        result.gate_passed = None
        return result
    card = evaluate_inpaint(
        original,
        result.output_path,
        mask_img,
        prompt,
        run_vlm=True,
        api_key=api_key,
    )
    result.score_card = card
    result.gate_passed = card.passed
    return result


def _run_sdxl(
    image,
    mask,
    prompt: str,
    *,
    negative_prompt: str | None,
    seed: int | None,
    parent_step_id: str | None,
    api_key: str | None,
    timeout: int,
) -> InpaintResult:
    return inpaint_sdxl.inpaint(
        image,
        mask,
        prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        parent_step_id=parent_step_id,
        api_key=api_key,
        timeout=timeout,
    )


def _run_kontext(
    image,
    mask,
    prompt: str,
    *,
    mode: KontextMode,
    seed: int | None,
    reference_image,
    parent_step_id: str | None,
    api_key: str | None,
    timeout: int,
) -> InpaintResult:
    return inpaint_kontext.inpaint(
        image,
        mask,
        prompt,
        mode=mode,
        seed=seed,
        reference_image=reference_image,
        parent_step_id=parent_step_id,
        api_key=api_key,
        timeout=timeout,
    )


def inpaint(
    image: Union[str, Path, bytes, Image.Image],
    mask: Union[str, Path, Image.Image, MaskInstance],
    prompt: str,
    *,
    backend: BackendName = "auto",
    negative_prompt: str | None = None,
    seed: int | None = None,
    reference_image: Union[str, Path, bytes, Image.Image] | None = None,
    parent_step_id: str | None = None,
    run_quality_gate: bool = True,
    kontext_mode: KontextMode = "masked",
    api_key: str | None = None,
    timeout: int = 180,
) -> InpaintResult:
    """
    Masked inpaint of `image` inside `mask` according to `prompt`.

    backend:
      - auto: surface → SDXL then Kontext on fail; structural → 2x mask expand + Kontext
      - sdxl / flux_kontext: forced (still score if run_quality_gate)
      - animeadapter: stub (raises InpaintError)
    """
    if backend not in VALID_BACKENDS:
        raise InpaintError(
            f"backend must be one of {VALID_BACKENDS}, got {backend!r}"
        )
    if not prompt or not str(prompt).strip():
        raise InpaintError("prompt is required")

    original = load_image(image)
    mask_img = normalize_mask(mask, original.size)

    if backend == "animeadapter":
        # Always raises
        return inpaint_animeadapter.inpaint(
            image, mask, prompt, parent_step_id=parent_step_id, api_key=api_key
        )

    if backend == "sdxl":
        result = _run_sdxl(
            image,
            mask_img,
            prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            parent_step_id=parent_step_id,
            api_key=api_key,
            timeout=timeout,
        )
        result = _attach_gate(
            result,
            original,
            mask_img,
            prompt,
            run_quality_gate=run_quality_gate,
            api_key=api_key,
        )
        result.routing_reason = "forced_sdxl"
        append_provenance(result)
        return result

    if backend == "flux_kontext":
        result = _run_kontext(
            image,
            mask_img,
            prompt,
            mode=kontext_mode,
            seed=seed,
            reference_image=reference_image,
            parent_step_id=parent_step_id,
            api_key=api_key,
            timeout=timeout,
        )
        # Instruction mode may not preserve outside-mask; still score for visibility
        result = _attach_gate(
            result,
            original,
            mask_img,
            prompt,
            run_quality_gate=run_quality_gate,
            api_key=api_key,
        )
        result.routing_reason = "forced_kontext"
        append_provenance(result)
        return result

    # ---- auto ----
    edit_type = classify_edit_type(prompt)

    # Structural / shape edits: skip SDXL (known-weak on this prompt class).
    # Expand mask 2x about centroid before Kontext. Clamping is image-edge only —
    # expansion can still bleed into neighboring characters/props in a busy panel;
    # a future character-bbox-aware check is needed once the Character Bible tracks
    # per-panel object positions. No prompt rewriting here (deferred to planner).
    if edit_type == "structural":
        bounds = original.size  # (width, height)
        clipped = mask_expansion_clipped(mask_img, factor=2.0, image_bounds=bounds)
        expanded = expand_mask(mask_img, factor=2.0, image_bounds=bounds)
        result = _run_kontext(
            image,
            expanded,
            prompt,
            mode="masked",
            seed=seed,
            reference_image=reference_image,
            parent_step_id=parent_step_id,
            api_key=api_key,
            timeout=timeout,
        )
        result.fallback_from = None
        result = _attach_gate(
            result,
            original,
            expanded,
            prompt,
            run_quality_gate=run_quality_gate,
            api_key=api_key,
        )
        result.routing_reason = "structural_direct_kontext_expanded_mask"
        result.mask_expansion_clipped = clipped
        append_provenance(result)
        return result

    # Surface edits: SDXL first, Kontext on gate fail / API error
    primary_error: Exception | None = None
    primary: InpaintResult | None = None

    try:
        primary = _run_sdxl(
            image,
            mask_img,
            prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            parent_step_id=parent_step_id,
            api_key=api_key,
            timeout=timeout,
        )
        primary = _attach_gate(
            primary,
            original,
            mask_img,
            prompt,
            run_quality_gate=run_quality_gate,
            api_key=api_key,
        )
        if not run_quality_gate or (primary.score_card and primary.score_card.passed):
            primary.routing_reason = "surface_sdxl_first"
            append_provenance(primary)
            return primary
        # gate failed → fall through to Kontext
    except InpaintError as e:
        primary_error = e

    try:
        fallback = _run_kontext(
            image,
            mask_img,
            prompt,
            mode="masked",  # auto always uses masked fill
            seed=seed,
            reference_image=reference_image,
            parent_step_id=parent_step_id,
            api_key=api_key,
            timeout=timeout,
        )
    except InpaintError as fallback_error:
        if primary is not None:
            # Kontext failed but we still have an SDXL result (gate failed)
            primary.fallback_from = None
            primary.routing_reason = "surface_sdxl_failed_kontext_error"
            append_provenance(primary)
            return primary
        raise InpaintError(
            f"All backends failed. SDXL: {primary_error}; "
            f"flux_kontext: {fallback_error}"
        ) from fallback_error

    if primary is None:
        # SDXL errored; only Kontext succeeded
        fallback.fallback_from = "sdxl"
        fallback = _attach_gate(
            fallback,
            original,
            mask_img,
            prompt,
            run_quality_gate=run_quality_gate,
            api_key=api_key,
        )
        fallback.routing_reason = "surface_sdxl_error_fallback_kontext"
        append_provenance(fallback)
        return fallback

    fallback.fallback_from = "sdxl"
    fallback = _attach_gate(
        fallback,
        original,
        mask_img,
        prompt,
        run_quality_gate=run_quality_gate,
        api_key=api_key,
    )

    # Both exist and primary failed the gate — pick better by weighted score
    if not run_quality_gate:
        fallback.routing_reason = "surface_sdxl_failed_fallback_kontext"
        append_provenance(fallback)
        return fallback

    p_score = primary.score_card.weighted_score() if primary.score_card else -1.0
    f_score = fallback.score_card.weighted_score() if fallback.score_card else -1.0

    if f_score >= p_score:
        fallback.routing_reason = "surface_sdxl_failed_fallback_kontext"
        append_provenance(fallback)
        return fallback

    # Prefer primary if somehow better despite gate fail (shouldn't happen often)
    primary.fallback_from = None
    primary.routing_reason = "surface_sdxl_failed_kept_sdxl"
    append_provenance(primary)
    return primary

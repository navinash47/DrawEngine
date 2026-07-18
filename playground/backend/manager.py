"""
manager.py — Routes text-grounded segmentation across backends.

Primary: SAM 3 (sam3.py)
Fallback: YOLO-World + SAM 2 (grounded_sam2.py)

Writes provenance exactly once per successful public segment() call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

from backend import grounded_sam2, sam3
from backend.segment import (
    SegmentationError,
    SegmentationResult,
    append_provenance,
    save_masks,
)

BackendName = Literal["auto", "sam3", "grounded_sam2"]
VALID_BACKENDS = ("auto", "sam3", "grounded_sam2")


def _run_backend(
    name: str,
    image: Union[str, Path, bytes],
    text_prompt: str,
    prob_threshold: float,
    api_key: str | None,
    timeout: int,
) -> SegmentationResult:
    if name == "sam3":
        return sam3.segment(
            image,
            text_prompt,
            prob_threshold=prob_threshold,
            api_key=api_key,
            timeout=timeout,
        )
    if name == "grounded_sam2":
        return grounded_sam2.segment(
            image,
            text_prompt,
            prob_threshold=prob_threshold,
            api_key=api_key,
            timeout=timeout,
        )
    raise SegmentationError(f"Unknown backend: {name}")


def segment(
    image: Union[str, Path, bytes],
    text_prompt: str,
    prob_threshold: float = 0.5,
    api_key: str | None = None,
    timeout: int = 60,
    backend: BackendName = "auto",
) -> SegmentationResult:
    """
    Segment every instance of `text_prompt` in `image`.

    Args:
        image: path to an image file, or raw image bytes.
        text_prompt: short noun phrase, e.g. "the hat".
        prob_threshold: confidence cutoff (0-1).
        api_key: overrides ROBOFLOW_API_KEY if provided.
        timeout: per-request timeout in seconds.
        backend: "auto" (SAM 3, then grounded_sam2 on error/empty),
                 "sam3", or "grounded_sam2".

    Returns:
        SegmentationResult. Logged to data/logs/segmentation_log.jsonl;
        full masks saved to data/masks/<request_id>.json.
    """
    if backend not in VALID_BACKENDS:
        raise SegmentationError(
            f"backend must be one of {VALID_BACKENDS}, got {backend!r}"
        )

    if backend != "auto":
        result = _run_backend(
            backend, image, text_prompt, prob_threshold, api_key, timeout
        )
        append_provenance(result)
        save_masks(result)
        return result

    primary_error: Exception | None = None
    try:
        result = _run_backend(
            "sam3", image, text_prompt, prob_threshold, api_key, timeout
        )
        if result.instances:
            append_provenance(result)
            save_masks(result)
            return result
        primary_error = SegmentationError(
            f"SAM 3 returned no instances for prompt {text_prompt!r}"
        )
    except SegmentationError as e:
        primary_error = e

    try:
        result = _run_backend(
            "grounded_sam2", image, text_prompt, prob_threshold, api_key, timeout
        )
    except SegmentationError as fallback_error:
        raise SegmentationError(
            f"All backends failed. SAM 3: {primary_error}; "
            f"grounded_sam2: {fallback_error}"
        ) from fallback_error

    append_provenance(result)
    save_masks(result)
    return result

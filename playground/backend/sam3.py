"""
sam3.py — Primary text-grounded segmentation backend (Roboflow hosted SAM 3).

Does not write provenance; manager.py logs once after a successful run.
"""

from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path
from typing import Union

import requests

from backend.segment import (
    ROBOFLOW_BASE,
    MaskInstance,
    SegmentationError,
    SegmentationResult,
    hash_image_bytes,
    load_image_bytes,
    polygon_to_bbox,
    resolve_api_key,
)

ENDPOINT = f"{ROBOFLOW_BASE}/sam3/concept_segment"
MODEL_VERSION_TAG = "sam3/sam3_final"


def segment(
    image: Union[str, Path, bytes],
    text_prompt: str,
    prob_threshold: float = 0.5,
    api_key: str | None = None,
    timeout: int = 60,
) -> SegmentationResult:
    """Segment every instance of `text_prompt` via SAM 3 Promptable Concept Segmentation."""
    key = resolve_api_key(api_key)
    image_bytes = load_image_bytes(image)
    image_hash = hash_image_bytes(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "format": "polygon",
        "image": {"type": "base64", "value": b64},
        "prompts": [{"type": "text", "text": text_prompt}],
        "output_prob_thresh": prob_threshold,
    }

    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    try:
        resp = requests.post(
            f"{ENDPOINT}?api_key={key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise SegmentationError(f"SAM 3 API request failed: {e}") from e

    if resp.status_code != 200:
        raise SegmentationError(
            f"SAM 3 API returned {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()

    try:
        prompt_result = data["prompt_results"][0]
        predictions = prompt_result.get("predictions", [])
    except (KeyError, IndexError) as e:
        raise SegmentationError(f"Unexpected SAM 3 response shape: {data}") from e

    instances = []
    for idx, pred in enumerate(predictions):
        masks = pred.get("masks", [])
        if not masks:
            continue
        polygon = masks[0]
        instances.append(
            MaskInstance(
                instance_id=idx,
                confidence=pred.get("confidence", 0.0),
                polygon=polygon,
                bbox=polygon_to_bbox(polygon),
            )
        )

    return SegmentationResult(
        request_id=request_id,
        image_hash=image_hash,
        prompt=text_prompt,
        model_version=MODEL_VERSION_TAG,
        timestamp=t0,
        instances=instances,
        raw_response=data,
    )

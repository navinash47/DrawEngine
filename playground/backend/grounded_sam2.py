"""
grounded_sam2.py — Fallback text-grounded segmentation (YOLO-World → SAM 2).

Mirrors the classic Grounded-SAM 2 shape (open-vocab boxes, then box-prompted
masks) using Roboflow serverless endpoints so no local GPU is required.
True Grounding DINO is not on the Serverless Hosted API.

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

YOLO_WORLD_ENDPOINT = f"{ROBOFLOW_BASE}/yolo_world/infer"
SAM2_ENDPOINT = f"{ROBOFLOW_BASE}/sam2/segment_image"
MODEL_VERSION_TAG = "grounded_sam2/yolo_world+sam2_hiera_tiny"
YOLO_WORLD_VERSION = "v2-s"
SAM2_VERSION = "hiera_tiny"


def _yolo_world_detect(
    b64: str,
    text_prompt: str,
    confidence: float,
    api_key: str,
    timeout: int,
) -> list[dict]:
    """Return YOLO-World predictions (center-xywh boxes + confidence)."""
    payload = {
        "api_key": api_key,
        "image": {"type": "base64", "value": b64},
        "text": [text_prompt],
        "yolo_world_version_id": YOLO_WORLD_VERSION,
        "confidence": confidence,
    }

    try:
        resp = requests.post(
            YOLO_WORLD_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise SegmentationError(f"YOLO-World API request failed: {e}") from e

    if resp.status_code != 200:
        raise SegmentationError(
            f"YOLO-World API returned {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    predictions = data.get("predictions", [])
    if not isinstance(predictions, list):
        raise SegmentationError(f"Unexpected YOLO-World response shape: {data}")
    return predictions


def _sam2_segment_boxes(
    b64: str,
    boxes: list[dict],
    api_key: str,
    timeout: int,
) -> dict:
    """Prompt SAM 2 with center-xywh boxes; return raw JSON response."""
    # SAM 2 Box schema is center-x/y + width/height (see inference Sam2Prompt.Box).
    prompts = [
        {
            "box": {
                "x": float(b["x"]),
                "y": float(b["y"]),
                "width": float(b["width"]),
                "height": float(b["height"]),
            }
        }
        for b in boxes
    ]

    payload = {
        "api_key": api_key,
        "image": {"type": "base64", "value": b64},
        "prompts": {"prompts": prompts},
        "sam2_version_id": SAM2_VERSION,
        "format": "json",
        "multimask_output": False,
    }

    try:
        resp = requests.post(
            SAM2_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise SegmentationError(f"SAM 2 API request failed: {e}") from e

    if resp.status_code != 200:
        raise SegmentationError(
            f"SAM 2 API returned {resp.status_code}: {resp.text[:500]}"
        )

    return resp.json()


def _predictions_from_sam2(data: dict) -> list[dict]:
    """Normalize SAM 2 response to a list of {masks, confidence} preds."""
    if "predictions" in data and isinstance(data["predictions"], list):
        return data["predictions"]
    # Some deployments wrap like SAM 3
    if "prompt_results" in data:
        try:
            return data["prompt_results"][0].get("predictions", [])
        except (KeyError, IndexError, TypeError):
            pass
    raise SegmentationError(f"Unexpected SAM 2 response shape: {data}")


def segment(
    image: Union[str, Path, bytes],
    text_prompt: str,
    prob_threshold: float = 0.5,
    api_key: str | None = None,
    timeout: int = 60,
) -> SegmentationResult:
    """
    Grounded-SAM-2-style fallback: open-vocab detect (YOLO-World), then SAM 2 masks.
    """
    key = resolve_api_key(api_key)
    image_bytes = load_image_bytes(image)
    image_hash = hash_image_bytes(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    detections = _yolo_world_detect(
        b64, text_prompt, confidence=prob_threshold, api_key=key, timeout=timeout
    )

    # Keep only detections with usable boxes
    boxes = []
    box_confidences = []
    for det in detections:
        if not all(k in det for k in ("x", "y", "width", "height")):
            continue
        boxes.append(det)
        box_confidences.append(float(det.get("confidence", 0.0)))

    if not boxes:
        return SegmentationResult(
            request_id=request_id,
            image_hash=image_hash,
            prompt=text_prompt,
            model_version=MODEL_VERSION_TAG,
            timestamp=t0,
            instances=[],
            raw_response={"yolo_world": detections, "sam2": None},
        )

    sam2_data = _sam2_segment_boxes(b64, boxes, api_key=key, timeout=timeout)
    predictions = _predictions_from_sam2(sam2_data)

    instances = []
    for idx, pred in enumerate(predictions):
        masks = pred.get("masks", [])
        if not masks:
            continue
        polygon = masks[0]
        # Prefer SAM 2 mask confidence; fall back to grounding box confidence
        conf = pred.get("confidence")
        if conf is None and idx < len(box_confidences):
            conf = box_confidences[idx]
        instances.append(
            MaskInstance(
                instance_id=idx,
                confidence=float(conf or 0.0),
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
        raw_response={"yolo_world": detections, "sam2": sam2_data},
    )

"""
segment.py — Text-grounded segmentation primitive for the Playground.

Wraps Roboflow's hosted SAM 3 (Segment Anything 3) serverless API so we get
real SAM 3 masks without any local GPU / install. This is the backend
function ComicAgentEngine will eventually call directly — the Gradio UI
in ui/app.py is a thin layer on top of it, nothing else should import
requests/roboflow directly.

Env vars:
    ROBOFLOW_API_KEY   required. Get one free at https://app.roboflow.com
                       Loaded automatically from repo-root or playground `.env`.

Usage:
    from backend.segment import segment

    result = segment("path/to/panel.png", "the hat")
    for instance in result.instances:
        print(instance.confidence, instance.bbox)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROBOFLOW_ENDPOINT = "https://serverless.roboflow.com/sam3/concept_segment"
MODEL_VERSION_TAG = "sam3/sam3_final"  # logged for provenance; bump if Roboflow changes default

_PLAYGROUND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PLAYGROUND_ROOT.parent
# Prefer an already-exported shell var; otherwise pick up .env files.
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_PLAYGROUND_ROOT / ".env")

DATA_DIR = _PLAYGROUND_ROOT / "data"
MASKS_DIR = DATA_DIR / "masks"
LOGS_DIR = DATA_DIR / "logs"
PROVENANCE_LOG = LOGS_DIR / "segmentation_log.jsonl"

MASKS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class SegmentationError(RuntimeError):
    """Raised when the SAM 3 API call fails or returns something unusable."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MaskInstance:
    """One segmented instance matching the text prompt."""
    instance_id: int
    confidence: float
    polygon: list  # list of [x, y] points, image pixel coords
    bbox: dict  # {"x0","y0","x1","y1"} absolute pixel coords, derived from polygon


@dataclass
class SegmentationResult:
    """Full result of one segment() call, plus provenance metadata."""
    request_id: str
    image_hash: str
    prompt: str
    model_version: str
    timestamp: float
    instances: list = field(default_factory=list)  # list[MaskInstance]
    raw_response: dict = field(default_factory=dict)  # kept for debugging / re-derivation

    def to_log_dict(self) -> dict:
        """Provenance-log-friendly dict (no raw_response bulk, no numpy)."""
        return {
            "request_id": self.request_id,
            "image_hash": self.image_hash,
            "prompt": self.prompt,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "num_instances": len(self.instances),
            "instances": [
                {"instance_id": i.instance_id, "confidence": i.confidence, "bbox": i.bbox}
                for i in self.instances
            ],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def _load_image_bytes(image: Union[str, Path, bytes]) -> bytes:
    if isinstance(image, bytes):
        return image
    path = Path(image)
    if not path.exists():
        raise SegmentationError(f"Image path does not exist: {path}")
    return path.read_bytes()


def _polygon_to_bbox(polygon: list) -> dict:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}


def _append_provenance(result: SegmentationResult) -> None:
    with open(PROVENANCE_LOG, "a") as f:
        f.write(json.dumps(result.to_log_dict()) + "\n")


def _save_masks(result: SegmentationResult) -> Path:
    """Save full mask polygons (not just the log summary) to a per-request JSON file."""
    out_path = MASKS_DIR / f"{result.request_id}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                **result.to_log_dict(),
                "instances_full": [asdict(i) for i in result.instances],
            },
            f,
            indent=2,
        )
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment(
    image: Union[str, Path, bytes],
    text_prompt: str,
    prob_threshold: float = 0.5,
    api_key: str | None = None,
    timeout: int = 60,
) -> SegmentationResult:
    """
    Segment every instance of `text_prompt` in `image` using SAM 3.

    Args:
        image: path to an image file, or raw image bytes.
        text_prompt: short noun phrase, e.g. "the hat", "the girl in the red cloak".
                     SAM 3 wants simple noun phrases, not long referring expressions.
        prob_threshold: confidence cutoff (0-1) for returned instances.
        api_key: overrides ROBOFLOW_API_KEY env var if provided.
        timeout: request timeout in seconds.

    Returns:
        SegmentationResult with one MaskInstance per matched object, plus
        provenance metadata. Every call is logged to data/logs/segmentation_log.jsonl
        and full mask polygons are saved to data/masks/<request_id>.json.

    Raises:
        SegmentationError on missing API key, network failure, or empty/bad response.
    """
    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SegmentationError(
            "No Roboflow API key found. Set the ROBOFLOW_API_KEY env var "
            "(get a free key at https://app.roboflow.com) or pass api_key=."
        )

    image_bytes = _load_image_bytes(image)
    image_hash = _hash_image_bytes(image_bytes)
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
            f"{ROBOFLOW_ENDPOINT}?api_key={key}",
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
        # a prediction may contain multiple mask polygons (rare); take the first/primary
        masks = pred.get("masks", [])
        if not masks:
            continue
        polygon = masks[0]
        instances.append(
            MaskInstance(
                instance_id=idx,
                confidence=pred.get("confidence", 0.0),
                polygon=polygon,
                bbox=_polygon_to_bbox(polygon),
            )
        )

    result = SegmentationResult(
        request_id=request_id,
        image_hash=image_hash,
        prompt=text_prompt,
        model_version=MODEL_VERSION_TAG,
        timestamp=t0,
        instances=instances,
        raw_response=data,
    )

    _append_provenance(result)
    _save_masks(result)

    return result
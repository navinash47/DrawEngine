"""
segment.py — Shared types, helpers, and provenance for text-grounded segmentation.

Backends live in sam3.py / grounded_sam2.py; routing lives in manager.py.
Callers should keep importing from here:

    from backend.segment import segment

    result = segment("path/to/panel.png", "the hat")

Env vars:
    ROBOFLOW_API_KEY   required. Get one free at https://app.roboflow.com
                       Loaded automatically from repo-root or playground `.env`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PLAYGROUND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PLAYGROUND_ROOT.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_PLAYGROUND_ROOT / ".env")

ROBOFLOW_BASE = "https://serverless.roboflow.com"

DATA_DIR = _PLAYGROUND_ROOT / "data"
MASKS_DIR = DATA_DIR / "masks"
LOGS_DIR = DATA_DIR / "logs"
PROVENANCE_LOG = LOGS_DIR / "segmentation_log.jsonl"

MASKS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class SegmentationError(RuntimeError):
    """Raised when a segmentation backend fails or returns something unusable."""


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

def resolve_api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SegmentationError(
            "No Roboflow API key found. Set the ROBOFLOW_API_KEY env var "
            "(get a free key at https://app.roboflow.com) or pass api_key=."
        )
    return key


def hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def load_image_bytes(image: Union[str, Path, bytes]) -> bytes:
    if isinstance(image, bytes):
        return image
    path = Path(image)
    if not path.exists():
        raise SegmentationError(f"Image path does not exist: {path}")
    return path.read_bytes()


def polygon_to_bbox(polygon: list) -> dict:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}


def append_provenance(result: SegmentationResult) -> None:
    with open(PROVENANCE_LOG, "a") as f:
        f.write(json.dumps(result.to_log_dict()) + "\n")


def save_masks(result: SegmentationResult) -> Path:
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
# Public API (lazy re-export — manager owns routing + provenance writes)
# ---------------------------------------------------------------------------

def segment(*args, **kwargs) -> SegmentationResult:
    from backend.manager import segment as _segment

    return _segment(*args, **kwargs)

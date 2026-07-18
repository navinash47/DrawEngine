"""
inpaint.py — Shared types, fal auth, provenance, and public inpaint() API.

Backends live in inpaint_sdxl.py / inpaint_kontext.py / inpaint_animeadapter.py;
routing + quality-gate fallback live in inpaint_manager.py.

    from backend.inpaint import inpaint

    result = inpaint(image_path, mask, "a blue wizard hat", backend="auto")

Env vars:
    FAL_API_KEY or FAL_KEY   required. fal.ai API key.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

import requests
from dotenv import load_dotenv
from PIL import Image

from backend.mask_utils import load_mask
from backend.segment import MaskInstance, _PLAYGROUND_ROOT, _REPO_ROOT

load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_PLAYGROUND_ROOT / ".env")

DATA_DIR = _PLAYGROUND_ROOT / "data"
INPAINTS_DIR = DATA_DIR / "inpaints"
MASKS_OUT_DIR = DATA_DIR / "inpaint_masks"
LOGS_DIR = DATA_DIR / "logs"
PROVENANCE_LOG = LOGS_DIR / "inpaint_log.jsonl"

INPAINTS_DIR.mkdir(parents=True, exist_ok=True)
MASKS_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SDXL_MODEL = "fal-ai/fast-sdxl/inpainting"
KONTEXT_FILL_MODEL = "fal-ai/flux-pro/v1/fill"
KONTEXT_EDIT_MODEL = "fal-ai/flux-kontext/dev"
MOONDREAM_MODEL = "fal-ai/moondream2/visual-query"


class InpaintError(RuntimeError):
    """Raised when an inpaint backend fails or returns something unusable."""


@dataclass
class QualityScoreCard:
    outside_mask_fidelity: float  # 0..1 (1 = perfect preserve)
    inside_mask_changed: float  # 0..1 (1 = strong change)
    prompt_adherence: float  # 0..1 from VLM
    passed: bool
    reasons: list[str] = field(default_factory=list)
    raw_vlm: str | None = None

    def weighted_score(self) -> float:
        return (
            0.5 * self.outside_mask_fidelity
            + 0.2 * self.inside_mask_changed
            + 0.3 * self.prompt_adherence
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InpaintResult:
    request_id: str
    image_hash: str
    prompt: str
    negative_prompt: str | None
    seed: int | None
    backend: str
    model_version: str
    timestamp: float
    output_path: str
    mask_path: str | None = None
    parent_step_id: str | None = None
    fal_request_id: str | None = None
    fallback_from: str | None = None
    routing_reason: str | None = None
    gate_passed: bool | None = None
    score_card: QualityScoreCard | None = None
    mask_expansion_clipped: bool | None = None
    raw_response: dict = field(default_factory=dict)

    def to_log_dict(self) -> dict:
        d: dict[str, Any] = {
            "request_id": self.request_id,
            "image_hash": self.image_hash,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "backend": self.backend,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "output_path": self.output_path,
            "mask_path": self.mask_path,
            "parent_step_id": self.parent_step_id,
            "fal_request_id": self.fal_request_id,
            "fallback_from": self.fallback_from,
            "routing_reason": self.routing_reason,
            "gate_passed": self.gate_passed,
            "mask_expansion_clipped": self.mask_expansion_clipped,
            "score_card": self.score_card.to_dict() if self.score_card else None,
        }
        return d


def resolve_fal_key(api_key: str | None = None) -> str:
    """Resolve fal credentials; prefer explicit arg, then FAL_KEY, then FAL_API_KEY."""
    key = api_key or os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY")
    if not key:
        raise InpaintError(
            "No fal API key found. Set FAL_API_KEY or FAL_KEY in the repo-root "
            ".env (https://fal.ai/dashboard/keys) or pass api_key=."
        )
    # fal_client reads FAL_KEY
    os.environ["FAL_KEY"] = key
    return key


def hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def load_image(image: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, bytes):
        import io

        return Image.open(io.BytesIO(image)).convert("RGB")
    path = Path(image)
    if not path.exists():
        raise InpaintError(f"Image path does not exist: {path}")
    return Image.open(path).convert("RGB")


def new_request_id() -> str:
    return str(uuid.uuid4())


def ensure_fal_client():
    try:
        import fal_client
    except ImportError as e:
        raise InpaintError(
            "fal-client is not installed. Run: pip install fal-client"
        ) from e
    return fal_client


def upload_pil(image: Image.Image, suffix: str = ".png") -> str:
    """Upload a PIL image to fal storage; returns a public URL."""
    fal_client = ensure_fal_client()
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        image.save(tmp_path)
        return fal_client.upload_file(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def download_image(url: str, timeout: int = 120) -> Image.Image:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    import io

    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def extract_image_url(payload: dict) -> str:
    """Pull the first result image URL from a fal response dict."""
    if not payload:
        raise InpaintError("Empty fal response")
    images = payload.get("images")
    if images and isinstance(images, list) and images[0].get("url"):
        return images[0]["url"]
    image = payload.get("image")
    if isinstance(image, dict) and image.get("url"):
        return image["url"]
    if isinstance(image, str) and image.startswith("http"):
        return image
    raise InpaintError(f"No image URL in fal response keys={list(payload.keys())}")


def normalize_mask(
    mask: Union[str, Path, Image.Image, MaskInstance, "object"],
    size: tuple[int, int],
) -> Image.Image:
    import numpy as np

    from backend.mask_utils import instance_to_mask

    if isinstance(mask, MaskInstance):
        return instance_to_mask(size, mask)
    if isinstance(mask, np.ndarray):
        return load_mask(mask).resize(size, Image.NEAREST)
    img = load_mask(mask)
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    return img


def save_inpaint_output(result_image: Image.Image, request_id: str) -> Path:
    out_path = INPAINTS_DIR / f"{request_id}.png"
    result_image.save(out_path)
    return out_path


def save_mask_sidecar(mask: Image.Image, request_id: str) -> Path:
    out_path = MASKS_OUT_DIR / f"{request_id}.png"
    mask.convert("L").save(out_path)
    return out_path


def append_provenance(result: InpaintResult) -> None:
    with open(PROVENANCE_LOG, "a") as f:
        f.write(json.dumps(result.to_log_dict()) + "\n")


def composite_masked(
    original: Image.Image,
    result: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Force surgical edit: keep original pixels where mask is black,
    take result where mask is white. Soft masks blend via alpha.
    """
    orig = original.convert("RGB")
    res = result.convert("RGB")
    if res.size != orig.size:
        res = res.resize(orig.size, Image.LANCZOS)
    m = mask.convert("L")
    if m.size != orig.size:
        m = m.resize(orig.size, Image.NEAREST)
    return Image.composite(res, orig, m)


def subscribe_fal(model_id: str, arguments: dict, timeout: int = 180) -> tuple[dict, str | None]:
    """Call fal_client.subscribe; return (data dict, request_id). Logs estimated cost."""
    fal_client = ensure_fal_client()
    resolve_fal_key()
    try:
        handle = fal_client.subscribe(
            model_id,
            arguments=arguments,
            with_logs=False,
            client_timeout=timeout,
        )
    except Exception as e:
        raise InpaintError(f"fal call to {model_id} failed: {e}") from e

    # fal_client may return a dict directly or an object with .data / .request_id
    if isinstance(handle, dict):
        data, req_id = handle, None
    else:
        data = getattr(handle, "data", None) or handle
        req_id = getattr(handle, "request_id", None)
        if not isinstance(data, dict):
            try:
                data = dict(data)
            except Exception as e:
                raise InpaintError(f"Unexpected fal response type: {type(handle)}") from e

    try:
        from backend.cost_tracker import log_fal_call

        log_fal_call(model_id, arguments, data, request_id=req_id)
    except Exception:
        pass  # never block generation on cost logging

    return data, req_id


def inpaint(*args, **kwargs) -> InpaintResult:
    from backend.inpaint_manager import inpaint as _inpaint

    return _inpaint(*args, **kwargs)

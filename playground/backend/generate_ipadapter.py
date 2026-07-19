"""
generate_ipadapter.py — A3 IP-Adapter identity-conditioned txt2img.

Uses fal-ai/flux-general with XLabs FLUX IP-Adapter weights and a canonical
reference from character_bible.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.character_bible import (
    ART_STYLE_TOKEN,
    CharacterBibleError,
    get_character,
    has_reference,
    resolve_character_id,
)
from backend.identity_gate import IdentityScoreCard, evaluate_identity_match
from backend.inpaint import (
    DATA_DIR,
    LOGS_DIR,
    download_image,
    extract_image_url,
    load_image,
    new_request_id,
    resolve_fal_key,
    subscribe_fal,
    upload_pil,
)
IDENTITY_GENS_DIR = DATA_DIR / "identity_gens"
PROVENANCE_LOG = LOGS_DIR / "identity_gen_log.jsonl"
IDENTITY_GENS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FLUX_GENERAL_MODEL = "fal-ai/flux-general"
IP_ADAPTER_PATH_V1 = "XLabs-AI/flux-ip-adapter"
IP_ADAPTER_PATH_V2 = "XLabs-AI/flux-ip-adapter-v2"
IMAGE_ENCODER_PATH = "openai/clip-vit-large-patch14"


class IdentityGenError(RuntimeError):
    """Raised when identity generation fails."""


@dataclass
class GenerationResult:
    request_id: str
    character_id: str
    reference_image_path: str
    scene_prompt: str
    ip_adapter_strength: float
    seed: int | None
    output_path: str
    model_version: str
    timestamp: float
    fal_request_id: str | None = None
    full_prompt: str | None = None
    score_card: IdentityScoreCard | None = None
    gate_passed: bool | None = None
    raw_response: dict = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "character_id": self.character_id,
            "reference_image_path": self.reference_image_path,
            "scene_prompt": self.scene_prompt,
            "full_prompt": self.full_prompt,
            "ip_adapter_strength": self.ip_adapter_strength,
            "seed": self.seed,
            "output_path": self.output_path,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "fal_request_id": self.fal_request_id,
            "gate_passed": self.gate_passed,
            "score_card": self.score_card.to_dict() if self.score_card else None,
        }


def append_provenance(result: GenerationResult) -> None:
    with open(PROVENANCE_LOG, "a") as f:
        f.write(json.dumps(result.to_log_dict()) + "\n")


def _compose_prompt(name: str, short_description: str, scene_prompt: str) -> str:
    identity = name
    if short_description:
        identity = f"{name}, {short_description}"
    scene = scene_prompt.strip()
    return f"{identity}, {scene}, {ART_STYLE_TOKEN}"


def _ip_adapter_args(image_url: str, scale: float, path: str) -> dict:
    return {
        "path": path,
        "weight_name": "ip_adapter.safetensors",
        "image_encoder_path": IMAGE_ENCODER_PATH,
        "image_url": image_url,
        "scale": float(scale),
    }


def generate_with_identity(
    character_id: str,
    scene_prompt: str,
    ip_adapter_strength: float = 0.6,
    seed: int | None = None,
    *,
    run_identity_gate: bool = True,
    api_key: str | None = None,
) -> GenerationResult:
    """
    Generate a scene image conditioned on the character's canonical reference
    via fal-ai/flux-general + XLabs IP-Adapter.
    """
    if not scene_prompt or not scene_prompt.strip():
        raise IdentityGenError("scene_prompt is required")

    cid = resolve_character_id(character_id)
    if not has_reference(cid):
        raise IdentityGenError(
            f"no reference selected for '{cid}' — pick one in the "
            "Reference Picker tab first"
        )

    entry = get_character(cid)
    assert entry is not None
    ref_path = entry["reference_image_path"]
    name = entry.get("name") or cid
    short_desc = entry.get("short_description") or ""

    resolve_fal_key(api_key)
    request_id = new_request_id()
    full_prompt = _compose_prompt(name, short_desc, scene_prompt)
    ref_img = load_image(ref_path)
    ref_url = upload_pil(ref_img, suffix=".png")

    base_args: dict[str, Any] = {
        "prompt": full_prompt,
        "image_size": "square_hd",
        "num_images": 1,
        "use_real_cfg": True,
        "output_format": "png",
        "enable_safety_checker": True,
    }
    if seed is not None:
        base_args["seed"] = int(seed)

    model_version = f"{FLUX_GENERAL_MODEL}+{IP_ADAPTER_PATH_V1}"
    arguments = {
        **base_args,
        "ip_adapters": [
            _ip_adapter_args(ref_url, ip_adapter_strength, IP_ADAPTER_PATH_V1)
        ],
    }

    try:
        data, fal_req_id = subscribe_fal(FLUX_GENERAL_MODEL, arguments, timeout=300)
    except Exception as e1:
        # Fall back to v2 weights if v1 fails
        model_version = f"{FLUX_GENERAL_MODEL}+{IP_ADAPTER_PATH_V2}"
        arguments = {
            **base_args,
            "ip_adapters": [
                _ip_adapter_args(ref_url, ip_adapter_strength, IP_ADAPTER_PATH_V2)
            ],
        }
        try:
            data, fal_req_id = subscribe_fal(FLUX_GENERAL_MODEL, arguments, timeout=300)
        except Exception as e2:
            raise IdentityGenError(
                f"flux-general IP-Adapter failed (v1: {e1}; v2: {e2})"
            ) from e2

    url = extract_image_url(data)
    out_img = download_image(url)
    out_path = IDENTITY_GENS_DIR / f"{request_id}.png"
    out_img.save(out_path)

    used_seed = data.get("seed", seed)
    if isinstance(used_seed, list) and used_seed:
        used_seed = used_seed[0]
    if used_seed is not None:
        used_seed = int(used_seed)

    result = GenerationResult(
        request_id=request_id,
        character_id=cid,
        reference_image_path=str(ref_path),
        scene_prompt=scene_prompt.strip(),
        ip_adapter_strength=float(ip_adapter_strength),
        seed=used_seed,
        output_path=str(out_path),
        model_version=model_version,
        timestamp=time.time(),
        fal_request_id=fal_req_id,
        full_prompt=full_prompt,
        raw_response={k: data.get(k) for k in ("seed", "prompt", "timings") if k in data},
    )

    if run_identity_gate:
        card = evaluate_identity_match(
            ref_path,
            out_path,
            name,
            short_description=short_desc,
            request_id=request_id,
        )
        result.score_card = card
        result.gate_passed = card.passed

    append_provenance(result)
    return result

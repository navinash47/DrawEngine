"""
identity_gate.py — A3 identity consistency score card (Moondream2 yes/no).

Single-image VLM: ask whether the generated image shows the character
described in the Character Bible (fal Moondream does not take two images).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union

from PIL import Image

from backend.inpaint import (
    LOGS_DIR,
    MOONDREAM_MODEL,
    load_image,
    resolve_fal_key,
    subscribe_fal,
    upload_pil,
)
from backend.quality_gate import _parse_vlm_match

IDENTITY_GATE_LOG = LOGS_DIR / "identity_gate_log.jsonl"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class IdentityScoreCard:
    identity_match: int  # 0 or 1
    raw_vlm: str
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_identity_match(
    reference_image: Union[str, Path, bytes, Image.Image],
    generated_image: Union[str, Path, bytes, Image.Image],
    character_name: str,
    *,
    short_description: str = "",
    api_key: str | None = None,
    log: bool = True,
    request_id: str | None = None,
) -> IdentityScoreCard:
    """
    Score whether generated_image depicts character_name.

    reference_image is accepted for API symmetry / future multi-image VLMs;
    Moondream2 only receives the generated image plus a text description.
    """
    _ = load_image(reference_image)  # validate path exists
    gen = load_image(generated_image)
    resolve_fal_key(api_key)

    desc = short_description.strip()
    if desc:
        subject = f"{character_name}, {desc}"
    else:
        subject = character_name

    rubric = (
        f'Does this image show {subject}? '
        'Answer with only the word "yes" or the word "no".'
    )
    url = upload_pil(gen, suffix=".png")
    data, _ = subscribe_fal(
        MOONDREAM_MODEL,
        {"image_url": url, "prompt": rubric},
    )
    text = str(data.get("output") or data.get("text") or data.get("answer") or data)
    score, _ = _parse_vlm_match(text)
    match = 1 if score >= 0.5 else 0
    card = IdentityScoreCard(
        identity_match=match,
        raw_vlm=text[:500],
        passed=bool(match),
    )

    if log:
        entry: dict[str, Any] = {
            "request_id": request_id,
            "character_name": character_name,
            "short_description": short_description,
            "identity_match": card.identity_match,
            "raw_vlm": card.raw_vlm,
            "passed": card.passed,
            "timestamp": time.time(),
        }
        with open(IDENTITY_GATE_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    return card

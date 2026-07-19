"""
character_bible.py — Canonical character references for A3 identity conditioning.

Stores selected reference portraits in character_bible.json and generates
plain txt2img candidates (no IP-Adapter) for the Reference Picker UI.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from backend.inpaint import (
    download_image,
    extract_image_url,
    resolve_fal_key,
    subscribe_fal,
)
from backend.segment import _PLAYGROUND_ROOT

DATA_DIR = _PLAYGROUND_ROOT / "data"
CHARACTERS_DIR = DATA_DIR / "characters"
CANDIDATES_DIR = CHARACTERS_DIR / "candidates"
REFERENCES_DIR = CHARACTERS_DIR / "references"
BIBLE_PATH = CHARACTERS_DIR / "character_bible.json"

CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

FAST_SDXL_MODEL = "fal-ai/fast-sdxl"

ART_STYLE_TOKEN = (
    "warm painterly storybook illustration, soft brushwork, gentle lighting, "
    "educational history comic, illustrated character not photoreal"
)

CHARACTER_PRESETS: dict[str, dict[str, str]] = {
    "julius_caesar": {
        "name": "Julius Caesar",
        "short_description": (
            "Roman general and emperor with a defined aquiline profile, "
            "laurel wreath, and white toga"
        ),
        "portrait_prompt": (
            "Portrait of Julius Caesar, Roman general and emperor, middle-aged man "
            "with a defined aquiline profile, short cropped grey-brown hair, "
            "laurel wreath crown, white toga with purple border, calm authoritative "
            "expression, front-facing three-quarter view, plain soft warm background, "
            f"clear face, {ART_STYLE_TOKEN}"
        ),
    },
    "cleopatra": {
        "name": "Cleopatra",
        "short_description": (
            "Ptolemaic Egyptian queen with dark hair, ornate headdress, "
            "and regal golden jewelry"
        ),
        "portrait_prompt": (
            "Portrait of Cleopatra VII, Ptolemaic Egyptian queen, elegant woman "
            "with dark hair, ornate Egyptian headdress with uraeus cobra, gold "
            "jewelry and collar necklace, regal composed expression, front-facing "
            "three-quarter view, plain soft warm background, clear face, "
            f"{ART_STYLE_TOKEN}"
        ),
    },
}


class CharacterBibleError(RuntimeError):
    """Raised when character bible operations fail."""


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def resolve_character_id(character_name: str) -> str:
    """Map a display name or id to a bible key."""
    raw = character_name.strip()
    slug = _slug(raw)
    if slug in CHARACTER_PRESETS:
        return slug
    for cid, preset in CHARACTER_PRESETS.items():
        if preset["name"].lower() == raw.lower():
            return cid
    if slug:
        return slug
    raise CharacterBibleError(f"Unknown character: {character_name!r}")


def load_bible() -> dict[str, Any]:
    if not BIBLE_PATH.exists():
        return {}
    try:
        return json.loads(BIBLE_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise CharacterBibleError(f"Failed to read character bible: {e}") from e


def save_bible(data: dict[str, Any]) -> None:
    BIBLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BIBLE_PATH.write_text(json.dumps(data, indent=2) + "\n")


def list_characters() -> list[str]:
    """Return preset ids plus any extra keys already in the bible."""
    keys = list(CHARACTER_PRESETS.keys())
    for k in load_bible():
        if k not in keys:
            keys.append(k)
    return keys


def get_character(character_id: str) -> dict[str, Any] | None:
    return load_bible().get(character_id)


def has_reference(character_id: str) -> bool:
    entry = get_character(character_id)
    if not entry:
        return False
    path = entry.get("reference_image_path")
    return bool(path) and Path(path).exists()


def build_portrait_prompt(character_name: str, description: str | None = None) -> str:
    cid = resolve_character_id(character_name)
    preset = CHARACTER_PRESETS.get(cid)
    if description and description.strip():
        return (
            f"Portrait of {preset['name'] if preset else character_name}, "
            f"{description.strip()}, front-facing three-quarter view, "
            f"plain soft warm background, clear face, {ART_STYLE_TOKEN}"
        )
    if preset:
        return preset["portrait_prompt"]
    return (
        f"Portrait of {character_name}, front-facing three-quarter view, "
        f"plain soft warm background, clear face, {ART_STYLE_TOKEN}"
    )


def generate_reference_candidates(
    character_name: str,
    description: str | None = None,
    n: int = 4,
    *,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Generate N candidate reference images via plain txt2img (fal-ai/fast-sdxl).
    Returns list of {image_path, seed, prompt_used}.
    """
    resolve_fal_key(api_key)
    cid = resolve_character_id(character_name)
    prompt = build_portrait_prompt(character_name, description)
    out_dir = CANDIDATES_DIR / cid
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for i in range(n):
        seed = int(uuid.uuid4().int % (2**31 - 1))
        arguments = {
            "prompt": prompt,
            "image_size": "square_hd",
            "num_images": 1,
            "seed": seed,
            "enable_safety_checker": True,
        }
        data, _ = subscribe_fal(FAST_SDXL_MODEL, arguments, timeout=180)
        url = extract_image_url(data)
        img = download_image(url)
        # Prefer seed returned by fal when present
        used_seed = data.get("seed", seed)
        if isinstance(used_seed, list) and used_seed:
            used_seed = used_seed[0]
        fname = f"{cid}_cand_{i}_{used_seed}.png"
        path = out_dir / fname
        img.save(path)
        results.append(
            {
                "image_path": str(path),
                "seed": int(used_seed) if used_seed is not None else seed,
                "prompt_used": prompt,
            }
        )
    return results


def select_reference(
    character_id: str,
    source_path: str | Path,
    *,
    reference_seed: int | None = None,
    prompt_used: str | None = None,
) -> dict[str, Any]:
    """
    Copy source image into references/ and write/update character_bible.json.
    """
    cid = resolve_character_id(character_id)
    src = Path(source_path)
    if not src.exists():
        raise CharacterBibleError(f"Reference image not found: {src}")

    preset = CHARACTER_PRESETS.get(cid, {})
    dest = REFERENCES_DIR / f"{cid}.png"
    shutil.copy2(src, dest)

    entry = {
        "name": preset.get("name", cid.replace("_", " ").title()),
        "short_description": preset.get("short_description", ""),
        "reference_image_path": str(dest),
        "reference_seed": reference_seed,
        "prompt_used": prompt_used or preset.get("portrait_prompt", ""),
        "selected_at": time.time(),
        "source_path": str(src),
    }
    bible = load_bible()
    bible[cid] = entry
    save_bible(bible)
    return entry


def select_reference_from_candidate(
    character_id: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return select_reference(
        character_id,
        candidate["image_path"],
        reference_seed=candidate.get("seed"),
        prompt_used=candidate.get("prompt_used"),
    )


def list_candidate_paths(character_id: str) -> list[str]:
    cid = resolve_character_id(character_id)
    folder = CANDIDATES_DIR / cid
    if not folder.exists():
        return []
    return sorted(str(p) for p in folder.glob("*.png"))

"""
edit_classifier.py — Lightweight prompt heuristic for inpaint routing.

Returns 'structural' (shape/object change → prefer Kontext) or 'surface'
(color/material/texture → try SDXL first).
"""

from __future__ import annotations

import re
from typing import Literal

EditType = Literal["structural", "surface"]

# Size *change* words only — bare "small"/"large" stay surface-friendly
# so "Make it small green scarf" is not treated as structural.
_SIZE_CHANGE = (
    "smaller",
    "bigger",
    "larger",
    "shrink",
    "resize",
    "enlarge",
)

_COLORS = (
    "red",
    "blue",
    "green",
    "orange",
    "yellow",
    "purple",
    "pink",
    "black",
    "white",
    "brown",
    "gray",
    "grey",
    "gold",
    "silver",
    "cyan",
    "magenta",
    "beige",
    "crimson",
    "scarlet",
)

_MATERIALS = (
    "wooden",
    "wood",
    "metal",
    "metallic",
    "leather",
    "silk",
    "cotton",
    "wool",
    "steel",
    "iron",
    "bronze",
    "plastic",
    "glass",
    "fabric",
    "cloth",
    "knitted",
    "woven",
)

_TEXTURES = (
    "striped",
    "shiny",
    "matte",
    "glossy",
    "patterned",
    "checkered",
    "plaid",
    "dotted",
    "textured",
    "smooth",
    "rough",
    "solid",
)

_OBJECT_SWAP = (
    "rifle",
    "sword",
    "gun",
    "pistol",
    "helmet",
    "shield",
    "hat",
    "axe",
    "bow",
    "spear",
    "dagger",
    "blade",
    "weapon",
)

# Stopwords / verbs that are not concrete nouns for the fallback check
_STOP = {
    "a",
    "an",
    "the",
    "it",
    "to",
    "make",
    "made",
    "into",
    "with",
    "and",
    "or",
    "of",
    "on",
    "in",
    "for",
    "as",
    "be",
    "is",
    "are",
    "was",
    "were",
    "this",
    "that",
    "more",
    "less",
    "very",
    "just",
    "please",
    "change",
    "replace",
    "turn",
    "add",
    "remove",
}


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(_has_word(text, w) for w in words)


def classify_edit_type(prompt: str) -> EditType:
    """Returns 'structural' or 'surface' for auto-routing."""
    if not prompt or not str(prompt).strip():
        return "surface"

    text = str(prompt).strip().lower()

    # 1) Explicit size/shape change
    if _has_any(text, _SIZE_CHANGE):
        return "structural"

    # 2) Color / material / texture cues → surface (wins over object nouns)
    if (
        _has_any(text, _COLORS)
        or _has_any(text, _MATERIALS)
        or _has_any(text, _TEXTURES)
    ):
        return "surface"

    # 3) Known object-swap nouns without surface cues
    if _has_any(text, _OBJECT_SWAP):
        return "structural"

    # 4) Remaining concrete noun (alphabetic tokens not in stop list)
    tokens = re.findall(r"[a-z]+", text)
    nouns = [t for t in tokens if t not in _STOP and len(t) > 2]
    if nouns:
        return "structural"

    # 5) Safe default
    return "surface"


if __name__ == "__main__":
    samples = [
        "Make it small green scarf",
        "green scarf",
        "blue scarf",
        "orange scarf",
        "Smaller sword",
        "Rifle",
        "shiny armor",
        "striped shirt",
    ]
    for s in samples:
        print(f"{classify_edit_type(s):12}  {s!r}")

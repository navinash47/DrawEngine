"""
inpaint_animeadapter.py — Stub until AnimeAdapter weights / host exist.

AnimeAdapter (arXiv 2605.20237) is research-stage; code/weights release
upon acceptance. Registered so the router schema is ready.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image

from backend.inpaint import InpaintError, InpaintResult
from backend.segment import MaskInstance


def inpaint(
    image: Union[str, Path, bytes, Image.Image],
    mask: Union[str, Path, Image.Image, MaskInstance] | None,
    prompt: str,
    **kwargs,
) -> InpaintResult:
    del image, mask, prompt, kwargs
    raise InpaintError(
        "AnimeAdapter not available yet — watch-list only "
        "(arXiv 2605.20237; weights release upon acceptance). "
        "Use backend='sdxl', 'flux_kontext', or 'auto'."
    )

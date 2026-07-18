"""
viz.py — Render segmentation masks as overlays on the original image.

Kept separate from the segmentation backends because "how a mask looks" is a
UI concern, not a segmentation concern. ComicAgentEngine will use raw polygons
from segment() directly and never touch this module.
"""

from __future__ import annotations

from typing import Union
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from backend.segment import SegmentationResult

# distinct colors so multiple instances are easy to tell apart by eye
_PALETTE = [
    (255, 99, 71, 110),   # tomato
    (65, 105, 225, 110),  # royal blue
    (60, 179, 113, 110),  # medium sea green
    (255, 215, 0, 110),   # gold
    (218, 112, 214, 110), # orchid
    (255, 140, 0, 110),   # dark orange
]


def render_overlay(
    image: Union[str, Path, bytes],
    result: SegmentationResult,
    selected_instance_id: int | None = None,
) -> Image.Image:
    """
    Draw all mask polygons from `result` on top of `image` as translucent fills.

    If selected_instance_id is given, only that instance is drawn at full
    opacity/highlight and the rest are dimmed — used when a user is picking
    a specific instance out of several matches.

    Returns a PIL Image (RGBA) ready to display in Gradio.
    """
    if isinstance(image, (str, Path)):
        base = Image.open(image).convert("RGBA")
    else:
        import io
        base = Image.open(io.BytesIO(image)).convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for inst in result.instances:
        color = _PALETTE[inst.instance_id % len(_PALETTE)]
        if selected_instance_id is not None and inst.instance_id != selected_instance_id:
            # dim non-selected instances
            color = (color[0], color[1], color[2], 40)
        polygon_pts = [tuple(p) for p in inst.polygon]
        if len(polygon_pts) >= 3:
            draw.polygon(polygon_pts, fill=color, outline=(255, 255, 255, 200))
            # label with instance id + confidence near the bbox top-left
            label_pos = (inst.bbox["x0"], max(0, inst.bbox["y0"] - 14))
            draw.text(
                label_pos,
                f"#{inst.instance_id} ({inst.confidence:.2f})",
                fill=(255, 255, 255, 255),
            )

    return Image.alpha_composite(base, overlay)


def crop_to_instance(
    image: Union[str, Path, bytes],
    result: SegmentationResult,
    instance_id: int,
    padding: int = 20,
) -> Image.Image:
    """Crop the base image to the bbox of one instance, with padding. Handy for
    quick visual QA of a single segmented region."""
    if isinstance(image, (str, Path)):
        base = Image.open(image).convert("RGBA")
    else:
        import io
        base = Image.open(io.BytesIO(image)).convert("RGBA")

    inst = next((i for i in result.instances if i.instance_id == instance_id), None)
    if inst is None:
        raise ValueError(f"No instance with id {instance_id}")

    x0 = max(0, inst.bbox["x0"] - padding)
    y0 = max(0, inst.bbox["y0"] - padding)
    x1 = min(base.width, inst.bbox["x1"] + padding)
    y1 = min(base.height, inst.bbox["y1"] + padding)

    return base.crop((x0, y0, x1, y1))
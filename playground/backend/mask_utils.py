"""
mask_utils.py — Convert segmentation polygons into binary inpaint masks.

Convention (fal / SDXL): white (255) = edit region, black (0) = keep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from backend.segment import MaskInstance


def polygon_to_mask(
    size: tuple[int, int],
    polygon: list,
    *,
    feather: int = 0,
    dilate: int = 0,
) -> Image.Image:
    """
    Rasterize a polygon into an L-mode mask matching `size` (width, height).

    feather: Gaussian blur radius after fill (soft edges).
    dilate: expand the filled region by this many pixels before feathering.
    """
    width, height = size
    mask = Image.new("L", (width, height), 0)
    pts = [tuple(p) for p in polygon]
    if len(pts) < 3:
        return mask

    draw = ImageDraw.Draw(mask)
    draw.polygon(pts, fill=255)

    if dilate > 0:
        # MaxFilter expands bright regions; size must be odd.
        k = dilate * 2 + 1
        mask = mask.filter(ImageFilter.MaxFilter(k))

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    return mask


def instance_to_mask(
    size: tuple[int, int],
    instance: MaskInstance,
    *,
    feather: int = 2,
    dilate: int = 1,
) -> Image.Image:
    """Build a binary (soft) mask for one MaskInstance."""
    return polygon_to_mask(
        size,
        instance.polygon,
        feather=feather,
        dilate=dilate,
    )


def load_mask(mask: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """Normalize various mask inputs to an L-mode PIL Image."""
    if isinstance(mask, Image.Image):
        return mask.convert("L")
    if isinstance(mask, np.ndarray):
        arr = mask
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
        return Image.fromarray(arr).convert("L")
    path = Path(mask)
    if not path.exists():
        raise FileNotFoundError(f"Mask path does not exist: {path}")
    return Image.open(path).convert("L")


def mask_bbox(mask: Image.Image, padding: int = 8) -> tuple[int, int, int, int]:
    """Tight bbox around non-zero mask pixels, with padding, clipped to image."""
    arr = np.array(mask.convert("L"))
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return 0, 0, mask.width, mask.height
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(mask.width, int(xs.max()) + 1 + padding)
    y1 = min(mask.height, int(ys.max()) + 1 + padding)
    return x0, y0, x1, y1


def save_mask(mask: Image.Image, path: Union[str, Path]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mask.convert("L").save(out)
    return out

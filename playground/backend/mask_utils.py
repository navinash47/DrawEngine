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


def _mask_centroid_and_bbox(
    mask: Image.Image,
) -> tuple[float, float, int, int, int, int] | None:
    """Return (cx, cy, x0, y0, x1, y1) for non-zero pixels, or None if empty."""
    arr = np.asarray(mask.convert("L"), dtype=np.float64)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None
    weights = arr[ys, xs]
    cx = float(np.average(xs, weights=weights))
    cy = float(np.average(ys, weights=weights))
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    return cx, cy, x0, y0, x1, y1


def _ideal_expanded_rect(
    cx: float,
    cy: float,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    factor: float,
) -> tuple[float, float, float, float]:
    """Unclamped expanded bbox about centroid (left, top, right, bottom)."""
    w = (x1 - x0) * factor
    h = (y1 - y0) * factor
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def expand_mask(
    mask: Image.Image,
    factor: float = 2.0,
    image_bounds: tuple[int, int] | None = None,
) -> Image.Image:
    """Expand a mask's bounding region by `factor`, centered on the mask centroid.

    Fills the expanded rectangle (unioned with the original mask) and clamps to
    image_bounds if provided, else the mask's own (width, height).

    NOTE: Only clamps to image edges. Expansion can still bleed into a neighboring
    character or prop in a busy panel; a future character-bbox-aware check is needed
    once the Character Bible tracks per-panel object positions.
    """
    m = mask.convert("L")
    if factor <= 1.0:
        return m.copy()
    info = _mask_centroid_and_bbox(m)
    if info is None:
        return m.copy()
    cx, cy, x0, y0, x1, y1 = info
    width, height = image_bounds if image_bounds is not None else m.size
    ix0, iy0, ix1, iy1 = _ideal_expanded_rect(cx, cy, x0, y0, x1, y1, factor)
    nx0 = int(max(0, ix0))
    ny0 = int(max(0, iy0))
    nx1 = int(min(width, ix1))
    ny1 = int(min(height, iy1))
    out = Image.new("L", m.size, 0)
    if nx1 > nx0 and ny1 > ny0:
        draw = ImageDraw.Draw(out)
        draw.rectangle([nx0, ny0, nx1 - 1, ny1 - 1], fill=255)
    arr_m = np.asarray(m, dtype=np.uint8)
    arr_o = np.asarray(out, dtype=np.uint8)
    return Image.fromarray(np.maximum(arr_m, arr_o), mode="L")


def mask_expansion_clipped(
    mask: Image.Image,
    factor: float = 2.0,
    image_bounds: tuple[int, int] | None = None,
    clip_threshold: float = 0.30,
) -> bool:
    """True if clamping removes more than `clip_threshold` of expansion on any side."""
    if factor <= 1.0:
        return False
    m = mask.convert("L")
    info = _mask_centroid_and_bbox(m)
    if info is None:
        return False
    cx, cy, x0, y0, x1, y1 = info
    width, height = image_bounds if image_bounds is not None else m.size
    ix0, iy0, ix1, iy1 = _ideal_expanded_rect(cx, cy, x0, y0, x1, y1, factor)
    # Per-side expansion distance (ideal) vs lost to clamping
    left_exp = max(0.0, x0 - ix0)
    right_exp = max(0.0, ix1 - x1)
    top_exp = max(0.0, y0 - iy0)
    bottom_exp = max(0.0, iy1 - y1)
    left_clip = max(0.0, -ix0)
    right_clip = max(0.0, ix1 - width)
    top_clip = max(0.0, -iy0)
    bottom_clip = max(0.0, iy1 - height)

    def _frac(clipped: float, expanded: float) -> float:
        if expanded <= 0:
            return 0.0
        return clipped / expanded

    return (
        _frac(left_clip, left_exp) > clip_threshold
        or _frac(right_clip, right_exp) > clip_threshold
        or _frac(top_clip, top_exp) > clip_threshold
        or _frac(bottom_clip, bottom_exp) > clip_threshold
    )


def save_mask(mask: Image.Image, path: Union[str, Path]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mask.convert("L").save(out)
    return out

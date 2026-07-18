"""
repro_blue_scarf.py — Force SDXL on 'blue scarf' with multiple seeds.

Data-only: reports inside_mask_changed + prompt_adherence per seed.
Does not change routing or prompts.

Usage (from playground/):
    python -m backend.repro_blue_scarf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PLAYGROUND = Path(__file__).resolve().parent.parent
if str(_PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(_PLAYGROUND))

from backend.inpaint import inpaint
from backend.mask_utils import load_mask

DEFAULT_ORIGINAL = _PLAYGROUND / "data" / "_a2_smoke" / "wp13092590.jpg"
DEFAULT_MASK = (
    _PLAYGROUND
    / "data"
    / "inpaint_masks"
    / "1c714b3c-1579-4ec6-bdc4-6e892cb7cba8.png"
)
DEFAULT_SEEDS = (11, 22, 33)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reproduce blue-scarf SDXL across seeds")
    p.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    p.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    p.add_argument("--prompt", type=str, default="blue scarf")
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds",
    )
    args = p.parse_args(argv)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not args.original.exists() or not args.mask.exists():
        print(f"Missing original or mask: {args.original} / {args.mask}")
        return 1

    mask = load_mask(args.mask)
    print(f"prompt={args.prompt!r}  backend=sdxl  seeds={seeds}")
    print("-" * 72)

    for seed in seeds:
        try:
            result = inpaint(
                args.original,
                mask,
                args.prompt,
                backend="sdxl",
                seed=seed,
                run_quality_gate=True,
                parent_step_id="blue_scarf_repro",
            )
        except Exception as e:
            print(f"seed={seed}  ERROR: {e}")
            continue
        sc = result.score_card
        if sc is None:
            print(f"seed={seed}  no score_card")
            continue
        print(
            f"seed={seed}  inside_changed={sc.inside_mask_changed:.3f}  "
            f"prompt_adherence={sc.prompt_adherence:.3f}  "
            f"raw_vlm={sc.raw_vlm!r}  passed={sc.passed}  "
            f"request_id={result.request_id}"
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

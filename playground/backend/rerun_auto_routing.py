"""
rerun_auto_routing.py — Re-run logged prompts through auto router.

Uses Mikasa original + each log row's mask. Confirms structural prompts
skip SDXL and surface prompts still try SDXL first.

Usage (from playground/):
    python -m backend.rerun_auto_routing --dry-run
    python -m backend.rerun_auto_routing
    python -m backend.rerun_auto_routing --prompt-substr Rifle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PLAYGROUND = Path(__file__).resolve().parent.parent
if str(_PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(_PLAYGROUND))

from backend.cost_tracker import COST_LOG
from backend.edit_classifier import classify_edit_type
from backend.inpaint import PROVENANCE_LOG, SDXL_MODEL, inpaint
from backend.mask_utils import load_mask

DEFAULT_ORIGINAL = _PLAYGROUND / "data" / "_a2_smoke" / "wp13092590.jpg"


def _load_log_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _cost_log_len() -> int:
    if not COST_LOG.exists():
        return 0
    return sum(1 for line in COST_LOG.read_text().splitlines() if line.strip())


def _sdxl_calls_since(start_len: int) -> int:
    if not COST_LOG.exists():
        return 0
    lines = [ln for ln in COST_LOG.read_text().splitlines() if ln.strip()]
    new = lines[start_len:]
    n = 0
    for ln in new:
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if row.get("model") == SDXL_MODEL:
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Re-run auto routing on logged inpaints")
    p.add_argument("--log", type=Path, default=PROVENANCE_LOG)
    p.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    p.add_argument("--request-id", type=str, default=None)
    p.add_argument("--prompt-substr", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print classify_edit_type for each prompt (no API calls)",
    )
    args = p.parse_args(argv)

    rows = _load_log_rows(args.log)
    if args.request_id:
        rows = [r for r in rows if r.get("request_id") == args.request_id]
    if args.prompt_substr:
        sub = args.prompt_substr.lower()
        rows = [r for r in rows if sub in str(r.get("prompt") or "").lower()]
    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        print("No matching log rows.")
        return 1

    print(f"{'edit_type':12}  prompt")
    print("-" * 60)
    for row in rows:
        prompt = row.get("prompt") or ""
        print(f"{classify_edit_type(prompt):12}  {prompt!r}")

    if args.dry_run:
        return 0

    if not args.original.exists():
        print(f"Original not found: {args.original}")
        return 1

    print("\nLive auto re-runs:")
    print("=" * 72)

    for row in rows:
        prompt = row.get("prompt") or ""
        mask_path = Path(row.get("mask_path") or "")
        edit_type = classify_edit_type(prompt)
        print(f"\nprompt={prompt!r}  classify={edit_type}")
        if not mask_path.exists():
            print(f"  SKIP missing mask {mask_path}")
            continue

        mask = load_mask(mask_path)
        start = _cost_log_len()
        try:
            result = inpaint(
                args.original,
                mask,
                prompt,
                backend="auto",
                run_quality_gate=True,
                parent_step_id=row.get("parent_step_id"),
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        sdxl_n = _sdxl_calls_since(start)
        sc = result.score_card
        print(f"  backend={result.backend}  routing_reason={result.routing_reason}")
        print(f"  fallback_from={result.fallback_from}  sdxl_api_calls={sdxl_n}")
        print(f"  gate_passed={result.gate_passed}  request_id={result.request_id}")
        if sc:
            print(
                f"  PA={sc.prompt_adherence:.3f}  inside={sc.inside_mask_changed:.3f}  "
                f"raw_vlm={sc.raw_vlm!r}"
            )

        if edit_type == "structural" and sdxl_n > 0:
            print("  WARNING: structural path still called SDXL")
        if edit_type == "structural" and result.routing_reason != "structural_direct_kontext_expanded_mask":
            print(f"  WARNING: expected structural_direct_kontext_expanded_mask")
        if edit_type == "surface" and sdxl_n < 1:
            print("  WARNING: surface path did not call SDXL")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

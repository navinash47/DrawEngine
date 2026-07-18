"""
rerun_vlm_gate.py — Re-run quality gate / VLM judge on logged inpaint outputs.

Does NOT call SDXL/Kontext. Default original image is the Mikasa panel used
for A2 smoke tests.

Usage (from playground/):
    python -m backend.rerun_vlm_gate
    python -m backend.rerun_vlm_gate --prompt-substr scarf
    python -m backend.rerun_vlm_gate --vlm-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PLAYGROUND = Path(__file__).resolve().parent.parent
if str(_PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(_PLAYGROUND))

from backend.inpaint import PROVENANCE_LOG, load_image
from backend.mask_utils import load_mask
from backend.quality_gate import _prompt_adherence_vlm, evaluate_inpaint

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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Re-run VLM quality gate on logged inpaints")
    p.add_argument(
        "--log",
        type=Path,
        default=PROVENANCE_LOG,
        help="Path to inpaint_log.jsonl",
    )
    p.add_argument(
        "--original",
        type=Path,
        default=DEFAULT_ORIGINAL,
        help="Original panel image for full evaluate_inpaint",
    )
    p.add_argument("--request-id", type=str, default=None, help="Filter by request_id")
    p.add_argument(
        "--prompt-substr",
        type=str,
        default=None,
        help="Case-insensitive substring filter on prompt",
    )
    p.add_argument("--limit", type=int, default=None, help="Max rows to evaluate")
    p.add_argument(
        "--vlm-only",
        action="store_true",
        help="Skip local MAE; only call Moondream prompt adherence",
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

    if not args.vlm_only and not args.original.exists():
        print(f"Original image not found: {args.original}")
        print("Pass --original PATH or use --vlm-only")
        return 1

    print(f"Re-judging {len(rows)} row(s); original={args.original}")
    print("-" * 72)

    for row in rows:
        rid = row.get("request_id")
        prompt = row.get("prompt") or ""
        backend = row.get("backend")
        out_path = Path(row.get("output_path") or "")
        mask_path = Path(row.get("mask_path") or "")
        old_sc = row.get("score_card") or {}
        old_pa = old_sc.get("prompt_adherence")

        print(f"\nrequest_id={rid}")
        print(f"  prompt={prompt!r}  backend={backend}")
        print(f"  old prompt_adherence={old_pa}  old raw_vlm={old_sc.get('raw_vlm')!r}")

        if not out_path.exists() or not mask_path.exists():
            print(f"  SKIP missing files: out={out_path.exists()} mask={mask_path.exists()}")
            continue

        try:
            if args.vlm_only:
                res = load_image(out_path)
                mask = load_mask(mask_path)
                score, raw = _prompt_adherence_vlm(res, mask, prompt)
                print(f"  NEW prompt_adherence={score:.3f}")
                print(f"  raw_vlm={raw!r}")
                print(
                    f"  (logged inside_mask_changed={old_sc.get('inside_mask_changed')})"
                )
            else:
                card = evaluate_inpaint(
                    args.original,
                    out_path,
                    mask_path,
                    prompt,
                    run_vlm=True,
                )
                print(f"  NEW outside_mask_fidelity={card.outside_mask_fidelity:.3f}")
                print(f"  NEW inside_mask_changed={card.inside_mask_changed:.3f}")
                print(f"  NEW prompt_adherence={card.prompt_adherence:.3f}")
                print(f"  NEW passed={card.passed}")
                print(f"  raw_vlm={card.raw_vlm!r}")
                if card.reasons:
                    print(f"  reasons={card.reasons}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

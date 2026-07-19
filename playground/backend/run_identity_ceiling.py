"""
run_identity_ceiling.py — Task 5 batch: IP-Adapter @ 0.6 across varied scenes.

Usage (from playground/):
    python -m backend.run_identity_ceiling
    python -m backend.run_identity_ceiling --generate-refs   # also gen+auto-select refs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow `python backend/run_identity_ceiling.py` from playground/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.character_bible import (
    CHARACTER_PRESETS,
    generate_reference_candidates,
    get_character,
    has_reference,
    list_candidate_paths,
    select_reference_from_candidate,
)
from backend.generate_ipadapter import generate_with_identity

SCENES: dict[str, list[str]] = {
    "julius_caesar": [
        "addressing the Roman senate",
        "on a battlefield",
        "in a garden at sunset",
        "standing before marble columns",
    ],
    "cleopatra": [
        "on her throne",
        "aboard a ceremonial barge",
        "in the palace at Alexandria",
        "walking through a moonlit garden",
    ],
}

STRENGTH = 0.6


def ensure_references(*, generate: bool) -> dict[str, list[dict]]:
    """Return candidates per character; optionally generate and auto-select first."""
    all_cands: dict[str, list[dict]] = {}
    for cid in SCENES:
        if generate or not list_candidate_paths(cid):
            print(f"Generating 4 reference candidates for {cid}...")
            cands = generate_reference_candidates(cid, n=4)
        else:
            paths = list_candidate_paths(cid)
            cands = [{"image_path": p, "seed": None, "prompt_used": None} for p in paths]
        all_cands[cid] = cands
        if not has_reference(cid):
            print(f"Auto-selecting first candidate for {cid}: {cands[0]['image_path']}")
            select_reference_from_candidate(cid, cands[0])
        else:
            entry = get_character(cid)
            print(f"Using existing reference for {cid}: {entry['reference_image_path']}")
    return all_cands


def run_batch() -> list[dict]:
    rows: list[dict] = []
    for cid, scenes in SCENES.items():
        for scene in scenes:
            print(f"Generating {cid} @ {STRENGTH}: {scene!r}...")
            result = generate_with_identity(
                cid,
                scene,
                ip_adapter_strength=STRENGTH,
                run_identity_gate=True,
            )
            sc = result.score_card
            row = {
                "character": cid,
                "scene_prompt": scene,
                "seed": result.seed,
                "identity_match": sc.identity_match if sc else None,
                "raw_vlm": sc.raw_vlm if sc else None,
                "passed": sc.passed if sc else None,
                "output_path": result.output_path,
                "request_id": result.request_id,
            }
            rows.append(row)
            print(
                f"  -> seed={row['seed']} match={row['identity_match']} "
                f"passed={row['passed']} vlm={row['raw_vlm']!r}"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 identity ceiling batch")
    parser.add_argument(
        "--generate-refs",
        action="store_true",
        help="Generate reference candidates and auto-select the first if none selected",
    )
    parser.add_argument(
        "--refs-only",
        action="store_true",
        help="Only generate/select references; skip scene batch",
    )
    args = parser.parse_args()

    cands = ensure_references(generate=args.generate_refs or args.refs_only)

    print("\n=== Reference candidates ===")
    for cid, items in cands.items():
        print(f"\n{CHARACTER_PRESETS[cid]['name']} ({cid}):")
        for i, c in enumerate(items):
            print(f"  [{i}] seed={c.get('seed')} path={c['image_path']}")
        entry = get_character(cid)
        print(f"  SELECTED: {entry['reference_image_path'] if entry else None}")

    if args.refs_only:
        return

    rows = run_batch()
    out = Path(__file__).resolve().parent.parent / "data" / "logs" / "identity_ceiling_results.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\n=== Results table ({len(rows)} rows) ===")
    print(
        f"{'character':<16} {'scene':<36} {'seed':>10} {'match':>5} {'passed':>6}"
    )
    for r in rows:
        print(
            f"{r['character']:<16} {r['scene_prompt']:<36} "
            f"{str(r['seed']):>10} {str(r['identity_match']):>5} {str(r['passed']):>6}"
        )
        print(f"  raw_vlm: {r['raw_vlm']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

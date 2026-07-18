"""
struct_edit_experiments.py — Ordered experiments on Rifle / Smaller sword.

Evidence only: mask expansion, prompt specificity, strength proxies.
Does NOT wire router changes.

Usage (from playground/):
    python -m backend.struct_edit_experiments
    python -m backend.struct_edit_experiments --stop-early
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

_PLAYGROUND = Path(__file__).resolve().parent.parent
if str(_PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(_PLAYGROUND))

from backend import inpaint_kontext, inpaint_sdxl
from backend.inpaint import (
    load_image,
    subscribe_fal,
    upload_pil,
    download_image,
    extract_image_url,
    composite_masked,
    save_inpaint_output,
    new_request_id,
)
from backend.mask_utils import load_mask, mask_bbox
from backend.quality_gate import evaluate_inpaint

ORIGINAL = _PLAYGROUND / "data" / "_a2_smoke" / "wp13092590.jpg"
MASK_SWORD = (
    _PLAYGROUND
    / "data"
    / "inpaint_masks"
    / "4c98ccc7-b87e-4f6f-b76f-b182fb1f4533.png"
)
MASK_RIFLE = (
    _PLAYGROUND
    / "data"
    / "inpaint_masks"
    / "31b40bae-d78d-4b6e-9676-bb41835b3c3e.png"
)
SEED = 42
PA_PASS = 0.6

PROMPTS = {
    "sword_bare": "Smaller sword",
    "sword_desc": "a short dagger, half the size of a longsword",
    "rifle_bare": "Rifle",
    "rifle_desc": "a wooden hunting rifle with a long barrel",
}


def expand_mask(mask: Image.Image, scale: float) -> Image.Image:
    """Expand white region by scaling bbox about its center; fill expanded rect."""
    if scale <= 1.0:
        return mask.convert("L")
    m = mask.convert("L")
    x0, y0, x1, y1 = mask_bbox(m, padding=0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0) * scale
    h = (y1 - y0) * scale
    nx0 = int(max(0, cx - w / 2))
    ny0 = int(max(0, cy - h / 2))
    nx1 = int(min(m.width, cx + w / 2))
    ny1 = int(min(m.height, cy + h / 2))
    out = Image.new("L", m.size, 0)
    draw = ImageDraw.Draw(out)
    draw.rectangle([nx0, ny0, nx1 - 1, ny1 - 1], fill=255)
    # Keep original mask union so we never shrink coverage
    import numpy as np

    arr_m = np.asarray(m, dtype=np.uint8)
    arr_o = np.asarray(out, dtype=np.uint8)
    out = Image.fromarray(np.maximum(arr_m, arr_o), mode="L")
    return out


def _row(
    exp: str,
    prompt: str,
    mask_scale: float,
    strength_or_mode: str,
    sc,
    request_id: str,
) -> dict:
    return {
        "exp": exp,
        "prompt": prompt,
        "mask_scale": mask_scale,
        "strength_or_mode": strength_or_mode,
        "inside_changed": sc.inside_mask_changed if sc else None,
        "prompt_adherence": sc.prompt_adherence if sc else None,
        "raw_vlm": sc.raw_vlm if sc else None,
        "passed": sc.passed if sc else None,
        "request_id": request_id,
    }


def _print_row(r: dict) -> None:
    pa = r["prompt_adherence"]
    ic = r["inside_changed"]
    pa_s = f"{pa:.3f}" if pa is not None else "n/a"
    ic_s = f"{ic:.3f}" if ic is not None else "n/a"
    print(
        f"{r['exp']:6} | {r['prompt'][:40]:40} | "
        f"mask={r['mask_scale']:.1f} | {r['strength_or_mode']:18} | "
        f"inside={ic_s} | PA={pa_s} | raw={r['raw_vlm']!r} | "
        f"pass={r['passed']} | id={r['request_id']}"
    )


def _gate(original, result_path, mask, prompt: str):
    return evaluate_inpaint(original, result_path, mask, prompt, run_vlm=True)


def run_kontext_fill(original, mask, prompt: str, seed: int):
    result = inpaint_kontext.inpaint(
        original,
        mask,
        prompt,
        mode="masked",
        seed=seed,
        parent_step_id="struct_exp",
    )
    sc = _gate(original, result.output_path, mask, prompt)
    return result, sc


def run_kontext_instruction(original, mask, prompt: str, seed: int):
    """Instruction mode ignores mask for API; gate still uses mask for metrics."""
    result = inpaint_kontext.inpaint(
        original,
        mask,
        prompt,
        mode="instruction",
        seed=seed,
        parent_step_id="struct_exp",
    )
    sc = _gate(original, result.output_path, mask, prompt)
    return result, sc


def run_sdxl(original, mask, prompt: str, seed: int, strength: float):
    result = inpaint_sdxl.inpaint(
        original,
        mask,
        prompt,
        seed=seed,
        strength=strength,
        parent_step_id="struct_exp",
    )
    sc = _gate(original, result.output_path, mask, prompt)
    return result, sc


def run_flux_pro_kontext(original, mask, prompt: str, seed: int):
    """One-shot higher-tier smoke: fal-ai/flux-pro/kontext (no mask on API)."""
    base = load_image(original)
    url = upload_pil(base, suffix=".png")
    args = {
        "prompt": prompt,
        "image_url": url,
        "seed": seed,
        "guidance_scale": 3.5,
        "num_images": 1,
        "output_format": "png",
        "safety_tolerance": "5",
    }
    data, _ = subscribe_fal("fal-ai/flux-pro/kontext", args)
    out_url = extract_image_url(data)
    result_img = download_image(out_url)
    if result_img.size != base.size:
        result_img = result_img.resize(base.size, Image.LANCZOS)
    # Soft-constrain to mask for fair local gate (surgical composite)
    result_img = composite_masked(base, result_img, mask)
    rid = new_request_id()
    out_path = save_inpaint_output(result_img, rid)
    sc = _gate(original, out_path, mask, prompt)
    return rid, sc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stop-early", action="store_true", default=True)
    p.add_argument("--no-stop-early", action="store_false", dest="stop_early")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    if not ORIGINAL.exists():
        print(f"Missing original: {ORIGINAL}")
        return 1
    for path in (MASK_SWORD, MASK_RIFLE):
        if not path.exists():
            print(f"Missing mask: {path}")
            return 1

    original = load_image(ORIGINAL)
    sword_base = load_mask(MASK_SWORD)
    rifle_base = load_mask(MASK_RIFLE)
    seed = args.seed
    rows: list[dict] = []
    passed_prompts: set[str] = set()

    def record(r: dict) -> bool:
        rows.append(r)
        _print_row(r)
        ok = (r["prompt_adherence"] or 0) >= PA_PASS
        if ok:
            passed_prompts.add(r["prompt"])
        return ok

    print("=" * 100)
    print("EXP 1 — Mask geometry")
    print("=" * 100)

    # Smaller sword: 1.0 / 1.5 / 2.0
    sword_best_scale = 1.0
    for scale in (1.0, 1.5, 2.0):
        mask = expand_mask(sword_base, scale)
        prompt = PROMPTS["sword_bare"]
        try:
            result, sc = run_kontext_fill(original, mask, prompt, seed)
            ok = record(
                _row("exp1", prompt, scale, "kontext_fill", sc, result.request_id)
            )
            if ok:
                sword_best_scale = scale
                if args.stop_early:
                    print(f"  >> stop-early: {prompt!r} passed at mask={scale}")
                    break
            else:
                # keep largest scale tried as best if none pass
                sword_best_scale = scale
        except Exception as e:
            print(f"exp1 sword scale={scale} ERROR: {e}")

    # Rifle: 1.0 and 2.0
    rifle_best_scale = 1.0
    for scale in (1.0, 2.0):
        mask = expand_mask(rifle_base, scale)
        prompt = PROMPTS["rifle_bare"]
        try:
            result, sc = run_kontext_fill(original, mask, prompt, seed)
            ok = record(
                _row("exp1", prompt, scale, "kontext_fill", sc, result.request_id)
            )
            if ok:
                rifle_best_scale = scale
                if args.stop_early:
                    print(f"  >> stop-early: {prompt!r} passed at mask={scale}")
                    break
            else:
                rifle_best_scale = scale
        except Exception as e:
            print(f"exp1 rifle scale={scale} ERROR: {e}")

    sword_done = PROMPTS["sword_bare"] in passed_prompts
    rifle_done = PROMPTS["rifle_bare"] in passed_prompts

    print("\n" + "=" * 100)
    print("EXP 2 — Prompt specificity")
    print("=" * 100)

    if not sword_done:
        for key, scale in (
            ("sword_bare", sword_best_scale),
            ("sword_desc", sword_best_scale),
        ):
            prompt = PROMPTS[key]
            if key == "sword_bare":
                # already ran bare at this scale in exp1; skip duplicate if scale matches a prior row
                already = any(
                    r["exp"] == "exp1"
                    and r["prompt"] == prompt
                    and r["mask_scale"] == scale
                    for r in rows
                )
                if already:
                    continue
            mask = expand_mask(sword_base, scale)
            try:
                result, sc = run_kontext_fill(original, mask, prompt, seed)
                ok = record(
                    _row("exp2", prompt, scale, "kontext_fill", sc, result.request_id)
                )
                if ok and args.stop_early:
                    sword_done = True
                    print(f"  >> stop-early: {prompt!r} passed")
                    break
            except Exception as e:
                print(f"exp2 sword {key} ERROR: {e}")
        sword_done = sword_done or PROMPTS["sword_desc"] in passed_prompts

    if not rifle_done:
        for key, scale in (
            ("rifle_bare", rifle_best_scale),
            ("rifle_desc", rifle_best_scale),
        ):
            prompt = PROMPTS[key]
            if key == "rifle_bare":
                already = any(
                    r["exp"] == "exp1"
                    and r["prompt"] == prompt
                    and r["mask_scale"] == scale
                    for r in rows
                )
                if already:
                    continue
            mask = expand_mask(rifle_base, scale)
            try:
                result, sc = run_kontext_fill(original, mask, prompt, seed)
                ok = record(
                    _row("exp2", prompt, scale, "kontext_fill", sc, result.request_id)
                )
                if ok and args.stop_early:
                    rifle_done = True
                    print(f"  >> stop-early: {prompt!r} passed")
                    break
            except Exception as e:
                print(f"exp2 rifle {key} ERROR: {e}")
        rifle_done = rifle_done or PROMPTS["rifle_desc"] in passed_prompts

    # Best prompt+mask for exp3
    def best_for(family: str, base_mask, scale: float) -> tuple[str, Image.Image, float]:
        if family == "sword":
            candidates = [
                r
                for r in rows
                if "sword" in r["prompt"].lower() or "dagger" in r["prompt"].lower()
            ]
        else:
            candidates = [r for r in rows if "rifle" in r["prompt"].lower()]
        if candidates:
            best = max(candidates, key=lambda r: r["prompt_adherence"] or 0)
            return (
                best["prompt"],
                expand_mask(base_mask, best["mask_scale"]),
                best["mask_scale"],
            )
        bare = PROMPTS["sword_bare"] if family == "sword" else PROMPTS["rifle_bare"]
        return bare, expand_mask(base_mask, scale), scale

    print("\n" + "=" * 100)
    print("EXP 3 — Strength / commitment proxies")
    print("NOTE: fal-ai/flux-pro/v1/fill has NO denoise/strength parameter.")
    print("=" * 100)

    need_exp3 = not (sword_done and rifle_done)
    if need_exp3:
        for family, base_mask, default_scale, done in (
            ("sword", sword_base, sword_best_scale, sword_done),
            ("rifle", rifle_base, rifle_best_scale, rifle_done),
        ):
            if done:
                continue
            prompt, mask, scale = best_for(family, base_mask, default_scale)
            print(f"\n-- {family}: prompt={prompt!r} mask_scale={scale}")

            # A: SDXL strength 0.95 vs 1.0
            for strength in (0.95, 1.0):
                try:
                    result, sc = run_sdxl(original, mask, prompt, seed, strength)
                    ok = record(
                        _row(
                            "exp3",
                            prompt,
                            scale,
                            f"sdxl_str={strength}",
                            sc,
                            result.request_id,
                        )
                    )
                    if ok:
                        passed_prompts.add(prompt)
                        if args.stop_early:
                            print(f"  >> stop-early on SDXL strength={strength}")
                            break
                except Exception as e:
                    print(f"exp3 sdxl strength={strength} ERROR: {e}")

            if (prompt in passed_prompts) and args.stop_early:
                continue

            # B: instruction mode with descriptive prompt
            desc = PROMPTS["sword_desc"] if family == "sword" else PROMPTS["rifle_desc"]
            try:
                result, sc = run_kontext_instruction(original, mask, desc, seed)
                ok = record(
                    _row(
                        "exp3",
                        desc,
                        scale,
                        "kontext_instruction",
                        sc,
                        result.request_id,
                    )
                )
                if ok:
                    passed_prompts.add(desc)
                    if args.stop_early:
                        continue
            except Exception as e:
                print(f"exp3 instruction ERROR: {e}")

            if any(p in passed_prompts for p in (prompt, desc)) and args.stop_early:
                continue

            # C: flux-pro/kontext smoke
            try:
                rid, sc = run_flux_pro_kontext(original, mask, desc, seed)
                record(
                    _row("exp3", desc, scale, "flux_pro_kontext", sc, rid)
                )
            except Exception as e:
                print(f"exp3 flux-pro/kontext ERROR: {e}")

    # ---- Summary ----
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    for r in rows:
        _print_row(r)

    any_pass = any((r["prompt_adherence"] or 0) >= PA_PASS for r in rows)
    mask_wins = any(
        r["exp"] == "exp1" and (r["prompt_adherence"] or 0) >= PA_PASS for r in rows
    )
    prompt_wins = any(
        r["exp"] == "exp2" and (r["prompt_adherence"] or 0) >= PA_PASS for r in rows
    )
    strength_wins = any(
        r["exp"] == "exp3" and (r["prompt_adherence"] or 0) >= PA_PASS for r in rows
    )

    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    if mask_wins and not prompt_wins and not strength_wins:
        rec = "(a) fix via mask geometry"
    elif prompt_wins:
        rec = "(b) fix via prompt specificity"
    elif strength_wins:
        rec = "(c) fix via strength/mode — apply for structural edits only in router"
    elif not any_pass:
        rec = (
            "(d) capability ceiling — fast-sdxl + flux-pro/v1/fill insufficient "
            "for full silhouette object replacement on this panel"
        )
        print(
            "Higher-tier options (not wired):\n"
            "  - fal-ai/flux-pro/kontext ~$0.04/image (instruction-style)\n"
            "  - fal-ai/flux-pro/kontext/max ~$0.08–0.11/image class\n"
            "  - two-step: fill empty (remove) then regen object in expanded mask\n"
            "  - current fill: $0.05/MP"
        )
    else:
        rec = "mixed — see table; prefer the cheapest passing recipe"

    print(f"Recommendation: {rec}")
    print(
        "\nPlayground operating rule (draft):\n"
        "  Structural silhouette swaps (Rifle / resize sword) are NOT solved by "
        "routing alone; require either an expanded-mask + descriptive-prompt + "
        "high-strength recipe if experiments pass, or a higher-tier backend "
        "(flux-pro/kontext or two-step remove→regen) — not fill defaults."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

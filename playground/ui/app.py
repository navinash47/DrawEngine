"""
app.py — Playground Gradio UI (A1 Segmentation + A2 Inpaint).

Run:
    # put ROBOFLOW_API_KEY and FAL_API_KEY in the repo-root .env, then:
    python ui/app.py

This file stays thin: logic lives in backend/*.py.
"""

import json
import re
import sys
from pathlib import Path

# allow `python ui/app.py` to import the sibling backend/ package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from PIL import Image

from backend.segment import (
    MASKS_DIR,
    MaskInstance,
    SegmentationError,
    SegmentationResult,
    segment,
)

try:
    from backend.viz import render_overlay
except ImportError:
    render_overlay = None

from backend.inpaint import InpaintError, inpaint
from backend.mask_utils import instance_to_mask
from backend.cost_tracker import (
    format_meter_markdown,
    recent_rows_for_table,
    reset_session,
    session_totals,
    summarize,
)


def _cost_line() -> str:
    s = session_totals()
    total = summarize()["total_usd"]
    return (
        f"est. session ${s['session_usd']:.4f} ({s['session_calls']} calls) · "
        f"all-time ${total:.4f}"
    )


def refresh_costs():
    s = summarize()
    return format_meter_markdown(s), recent_rows_for_table(40)


def reset_session_costs():
    reset_session()
    return refresh_costs()


def _load_latest_seg_for_prompt(text_prompt: str) -> SegmentationResult | None:
    if not text_prompt or not text_prompt.strip():
        return None
    prompt = text_prompt.strip()
    candidates = sorted(MASKS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        data = json.loads(c.read_text())
        if data.get("prompt") == prompt:
            instances = [MaskInstance(**inst) for inst in data["instances_full"]]
            return SegmentationResult(
                request_id=data["request_id"],
                image_hash=data["image_hash"],
                prompt=data["prompt"],
                model_version=data["model_version"],
                timestamp=data["timestamp"],
                instances=instances,
            )
    return None


def _parse_instance_id(choice_label: str | None) -> int | None:
    if not choice_label:
        return None
    m = re.search(r"#(\d+)", choice_label)
    return int(m.group(1)) if m else None


def _format_score_card(result) -> str:
    if result.score_card is None:
        return "quality gate: skipped"
    sc = result.score_card
    lines = [
        f"gate_passed={result.gate_passed}",
        f"  outside_mask_fidelity={sc.outside_mask_fidelity:.3f}",
        f"  inside_mask_changed={sc.inside_mask_changed:.3f}",
        f"  prompt_adherence={sc.prompt_adherence:.3f}",
        f"  weighted={sc.weighted_score():.3f}",
    ]
    if sc.reasons:
        lines.append("  reasons: " + "; ".join(sc.reasons))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A1 — Segment tab
# ---------------------------------------------------------------------------

def run_segmentation(image_path, text_prompt, prob_threshold, backend):
    if image_path is None:
        return None, "Upload an image first.", gr.update(choices=[], value=None)
    if not text_prompt or not text_prompt.strip():
        return None, "Type a text prompt, e.g. 'the hat'.", gr.update(choices=[], value=None)

    try:
        result = segment(
            image_path,
            text_prompt.strip(),
            prob_threshold=prob_threshold,
            backend=backend,
        )
    except SegmentationError as e:
        return None, f"⚠️ {e}", gr.update(choices=[], value=None)

    if not result.instances:
        return (
            None,
            f"No matches found for '{text_prompt}' via {result.model_version}. "
            "Try a different phrase or backend.",
            gr.update(choices=[], value=None),
        )

    overlay_img = render_overlay(image_path, result) if render_overlay else None
    choices = [f"#{i.instance_id} (conf {i.confidence:.2f})" for i in result.instances]
    status = (
        f"Found {len(result.instances)} instance(s) for '{text_prompt}'. "
        f"backend={result.model_version} · request_id={result.request_id} · "
        f"logged to data/logs/segmentation_log.jsonl"
    )
    return overlay_img, status, gr.update(choices=choices, value=choices[0] if choices else None)


def highlight_instance(image_path, text_prompt, choice_label):
    if image_path is None or not choice_label:
        return None
    instance_id = _parse_instance_id(choice_label)
    result = _load_latest_seg_for_prompt(text_prompt)
    if result is None or render_overlay is None:
        return None
    return render_overlay(image_path, result, selected_instance_id=instance_id)


# ---------------------------------------------------------------------------
# A2 — Inpaint tab
# ---------------------------------------------------------------------------

def run_inpaint_segment(image_path, text_prompt, prob_threshold, seg_backend):
    """Segment inside the Inpaint tab; returns overlay, mask preview, status, picker."""
    if image_path is None:
        return None, None, "Upload an image first.", gr.update(choices=[], value=None)
    if not text_prompt or not text_prompt.strip():
        return None, None, "Type what to edit, e.g. 'the hat'.", gr.update(choices=[], value=None)

    try:
        result = segment(
            image_path,
            text_prompt.strip(),
            prob_threshold=prob_threshold,
            backend=seg_backend,
        )
    except SegmentationError as e:
        return None, None, f"⚠️ {e}", gr.update(choices=[], value=None)

    if not result.instances:
        return (
            None,
            None,
            f"No matches for '{text_prompt}'. Try another phrase.",
            gr.update(choices=[], value=None),
        )

    overlay_img = render_overlay(image_path, result) if render_overlay else None
    inst = result.instances[0]
    base = Image.open(image_path)
    mask_preview = instance_to_mask(base.size, inst)
    choices = [f"#{i.instance_id} (conf {i.confidence:.2f})" for i in result.instances]
    status = (
        f"Segmented {len(result.instances)} instance(s). "
        f"seg={result.model_version} · parent_step_id={result.request_id}"
    )
    return (
        overlay_img,
        mask_preview,
        status,
        gr.update(choices=choices, value=choices[0]),
    )


def preview_inpaint_mask(image_path, text_prompt, choice_label):
    if image_path is None or not choice_label:
        return None
    instance_id = _parse_instance_id(choice_label)
    result = _load_latest_seg_for_prompt(text_prompt)
    if result is None or instance_id is None:
        return None
    inst = next((i for i in result.instances if i.instance_id == instance_id), None)
    if inst is None:
        return None
    base = Image.open(image_path)
    return instance_to_mask(base.size, inst)


def run_inpaint(
    image_path,
    seg_prompt,
    choice_label,
    edit_prompt,
    inpaint_backend,
    kontext_mode,
    run_gate,
    seed,
    negative_prompt,
):
    if image_path is None:
        return None, "Upload an image first."
    if not edit_prompt or not edit_prompt.strip():
        return None, "Type an edit prompt, e.g. 'a tall blue wizard hat'."
    if not choice_label:
        return None, "Segment first and pick an instance."

    result_seg = _load_latest_seg_for_prompt(seg_prompt)
    if result_seg is None:
        return None, "No saved segmentation for that phrase. Click Segment first."

    instance_id = _parse_instance_id(choice_label)
    inst = next((i for i in result_seg.instances if i.instance_id == instance_id), None)
    if inst is None:
        return None, f"Instance {choice_label} not found in last segmentation."

    seed_val = int(seed) if seed is not None and str(seed).strip() != "" else None
    neg = negative_prompt.strip() if negative_prompt and negative_prompt.strip() else None

    try:
        result = inpaint(
            image_path,
            inst,
            edit_prompt.strip(),
            backend=inpaint_backend,
            negative_prompt=neg,
            seed=seed_val,
            parent_step_id=result_seg.request_id,
            run_quality_gate=bool(run_gate),
            kontext_mode=kontext_mode,
        )
    except InpaintError as e:
        return None, f"⚠️ {e}"

    out_img = Image.open(result.output_path)
    status = (
        f"backend={result.backend} · model={result.model_version}\n"
        f"request_id={result.request_id} · fal={result.fal_request_id}\n"
        f"fallback_from={result.fallback_from} · parent={result.parent_step_id}\n"
        f"output={result.output_path}\n"
        f"{_format_score_card(result)}\n"
        f"{_cost_line()}"
    )
    return out_img, status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Operation Shustrutha — Playground") as demo:
    gr.Markdown("# The Playground")
    gr.Markdown(
        "A1 text-grounded segmentation (Roboflow) · "
        "A2 masked inpainting (fal SDXL → FLUX Kontext + quality gate)"
    )

    with gr.Tab("Segment"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload panel / image")
                prompt_input = gr.Textbox(
                    label="What to segment",
                    placeholder="e.g. the hat, the girl in the red cloak",
                )
                threshold_slider = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05, label="Confidence threshold"
                )
                backend_dropdown = gr.Dropdown(
                    choices=["auto", "sam3", "grounded_sam2"],
                    value="auto",
                    label="Backend",
                    info="auto = SAM 3, fall back to YOLO-World+SAM 2 on error/empty",
                )
                run_btn = gr.Button("Segment", variant="primary")
                status_box = gr.Textbox(label="Status", interactive=False)
                instance_picker = gr.Dropdown(
                    label="Instances found (pick to highlight)",
                    choices=[],
                    interactive=True,
                )

            with gr.Column(scale=1):
                output_image = gr.Image(label="Masks overlaid on image", type="pil")

        run_btn.click(
            fn=run_segmentation,
            inputs=[image_input, prompt_input, threshold_slider, backend_dropdown],
            outputs=[output_image, status_box, instance_picker],
        )
        instance_picker.change(
            fn=highlight_instance,
            inputs=[image_input, prompt_input, instance_picker],
            outputs=[output_image],
        )

    with gr.Tab("Inpaint"):
        with gr.Row():
            with gr.Column(scale=1):
                ip_image = gr.Image(type="filepath", label="Upload panel / image")
                ip_seg_prompt = gr.Textbox(
                    label="What to edit (segment)",
                    placeholder="e.g. the hat",
                )
                ip_threshold = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05, label="Segment confidence"
                )
                ip_seg_backend = gr.Dropdown(
                    choices=["auto", "sam3", "grounded_sam2"],
                    value="auto",
                    label="Segment backend",
                )
                ip_seg_btn = gr.Button("Segment", variant="secondary")
                ip_instance = gr.Dropdown(
                    label="Instance to inpaint",
                    choices=[],
                    interactive=True,
                )
                ip_edit_prompt = gr.Textbox(
                    label="Edit prompt",
                    placeholder="e.g. a tall blue wizard hat, comic style",
                )
                ip_negative = gr.Textbox(
                    label="Negative prompt (SDXL)",
                    placeholder="optional",
                )
                ip_backend = gr.Dropdown(
                    choices=["auto", "sdxl", "flux_kontext", "animeadapter"],
                    value="auto",
                    label="Inpaint backend",
                    info="auto = SDXL → quality gate → FLUX Kontext on fail",
                )
                ip_kontext_mode = gr.Dropdown(
                    choices=["masked", "instruction"],
                    value="masked",
                    label="Kontext mode (when backend=flux_kontext)",
                )
                ip_seed = gr.Number(label="Seed (optional)", precision=0, value=None)
                ip_gate = gr.Checkbox(label="Run quality gate", value=True)
                ip_run_btn = gr.Button("Inpaint", variant="primary")
                ip_status = gr.Textbox(label="Status / score card", interactive=False, lines=10)

            with gr.Column(scale=1):
                ip_overlay = gr.Image(label="Segment overlay", type="pil")
                ip_mask = gr.Image(label="Mask preview (white = edit)", type="pil")
                ip_result = gr.Image(label="Inpaint result", type="pil")

        ip_seg_btn.click(
            fn=run_inpaint_segment,
            inputs=[ip_image, ip_seg_prompt, ip_threshold, ip_seg_backend],
            outputs=[ip_overlay, ip_mask, ip_status, ip_instance],
        )
        ip_instance.change(
            fn=preview_inpaint_mask,
            inputs=[ip_image, ip_seg_prompt, ip_instance],
            outputs=[ip_mask],
        )
        ip_run_btn.click(
            fn=run_inpaint,
            inputs=[
                ip_image,
                ip_seg_prompt,
                ip_instance,
                ip_edit_prompt,
                ip_backend,
                ip_kontext_mode,
                ip_gate,
                ip_seed,
                ip_negative,
            ],
            outputs=[ip_result, ip_status],
        )

    with gr.Tab("Costs"):
        gr.Markdown(
            "Track **estimated** API spend (fal + Roboflow). "
            "Not invoices — rates are local approximations in `backend/cost_tracker.py`."
        )
        cost_meter = gr.Markdown(value=format_meter_markdown())
        cost_table = gr.Dataframe(
            headers=["time", "provider", "model", "operation", "est_usd", "request_id"],
            value=recent_rows_for_table(40),
            interactive=False,
            label="Recent API calls",
            wrap=True,
        )
        with gr.Row():
            cost_refresh = gr.Button("Refresh", variant="primary")
            cost_reset_session = gr.Button("Reset session counter")
        cost_refresh.click(fn=refresh_costs, inputs=[], outputs=[cost_meter, cost_table])
        cost_reset_session.click(
            fn=reset_session_costs, inputs=[], outputs=[cost_meter, cost_table]
        )

    with gr.Tab("About"):
        gr.Markdown(
            """
            **The Playground** (Operation Shustrutha) — composable image-editing primitives.

            ### A1 — Segmentation
            - Primary: `backend/sam3.py` (Roboflow SAM 3)
            - Fallback: `backend/grounded_sam2.py` (YOLO-World → SAM 2)
            - Router: `backend/manager.py` (`auto`)
            - Logs: `data/logs/segmentation_log.jsonl`, masks in `data/masks/`

            ### A2 — Masked inpainting
            - Primary: `fal-ai/fast-sdxl/inpainting`
            - Fallback: `fal-ai/flux-pro/v1/fill` (FLUX Kontext fill)
            - Stub: AnimeAdapter (watch-list until weights exist)
            - Quality gate: outside-mask fidelity + inside change + Moondream2 prompt check
            - Router: `backend/inpaint_manager.py` (`auto` = SDXL → gate → Kontext)
            - Logs: `data/logs/inpaint_log.jsonl`, outputs in `data/inpaints/`

            ### Costs
            - Estimated USD meter: Costs tab · log `data/logs/api_cost_log.jsonl`

            ### Env
            - `ROBOFLOW_API_KEY` — segmentation
            - `FAL_API_KEY` or `FAL_KEY` — inpainting + VLM gate

            ### Next
            - A3 identity (IP-Adapter / LoRA), A4 ControlNet stacking
            """
        )

if __name__ == "__main__":
    demo.launch()

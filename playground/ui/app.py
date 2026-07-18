"""
app.py — Playground Gradio UI, starting with the A1 Segmentation tab.

Run:
    # put ROBOFLOW_API_KEY in the repo-root .env (auto-loaded), then:
    python ui/app.py

This file should stay thin: all real logic lives in backend/segment.py and
backend/viz.py. The UI just wires user input to those functions and shows
the result.
"""

import sys
from pathlib import Path

# allow `python ui/app.py` to import the sibling backend/ package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from backend.segment import segment, SegmentationError

try:
    from backend.viz import render_overlay
except ImportError:
    render_overlay = None  # Pillow missing; overlay tab will show an error


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
    """Re-render the overlay with only the selected instance highlighted.
    Re-runs against the last saved mask file rather than re-calling the API."""
    if image_path is None or not choice_label:
        return None
    import json
    import re
    from backend.segment import MASKS_DIR, SegmentationResult, MaskInstance

    instance_id = int(re.search(r"#(\d+)", choice_label).group(1))

    # find the most recent mask file matching this prompt (simple approach for now)
    candidates = sorted(MASKS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        data = json.loads(c.read_text())
        if data["prompt"] == text_prompt.strip():
            instances = [
                MaskInstance(**inst) for inst in data["instances_full"]
            ]
            result = SegmentationResult(
                request_id=data["request_id"],
                image_hash=data["image_hash"],
                prompt=data["prompt"],
                model_version=data["model_version"],
                timestamp=data["timestamp"],
                instances=instances,
            )
            return render_overlay(image_path, result, selected_instance_id=instance_id)
    return None


with gr.Blocks(title="Operation Shustrutha — Playground") as demo:
    gr.Markdown("# 🎨 The Playground")
    gr.Markdown(
        "**A1 — Text-Grounded Segmentation** "
        "(SAM 3 primary, YOLO-World + SAM 2 fallback — both via Roboflow)"
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
                    label="Instances found (pick to highlight)", choices=[], interactive=True
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

    with gr.Tab("About"):
        gr.Markdown(
            """
            This is the first primitive of **The Playground** (Operation Shustrutha).

            - **Primary**: `backend/sam3.py` — Roboflow hosted SAM 3
              (`sam3/concept_segment`).
            - **Fallback**: `backend/grounded_sam2.py` — YOLO-World (text→boxes)
              then SAM 2 (boxes→masks). Same API key; no local GPU.
            - **Router**: `backend/manager.py` tries SAM 3 first in `auto` mode,
              then falls back on error or empty results. Provenance is written once.
            - Every call is logged to `data/logs/segmentation_log.jsonl` and full
              mask polygons are saved to `data/masks/<request_id>.json` — this is
              the seed of the edit-provenance system future tools (inpainting,
              LoRA, human-correction retrace) will build on.
            - Next primitives to add: masked inpainting (A2), identity
              conditioning (A3), ControlNet stacking (A4).
            """
        )

if __name__ == "__main__":
    demo.launch()

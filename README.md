# The Playground — Operation Shustrutha

Composable image-editing primitives, built backend-function-first so
ComicAgentEngine can call them directly later. The Gradio UI is a thin
layer for you (and beta testers) to drive them by eye.

## Status
- ✅ **A1 — Text-grounded segmentation** (SAM 3 primary + YOLO-World/SAM 2 fallback via Roboflow)
- ✅ **A2 — Masked inpainting** (fal SDXL primary → FLUX Kontext fallback + visual quality gate; AnimeAdapter stub)
- ⬜ A3 — Identity conditioning (IP-Adapter / LoRA)
- ⬜ A4 — ControlNet stacking

## Setup (on your Mac — no local GPU needed)

```bash
cd playground
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Put API keys in a repo-root `.env` file:

```bash
# DrawEngine/.env
ROBOFLOW_API_KEY=your_roboflow_key
FAL_API_KEY=your_fal_key
```

(`FAL_KEY` is also accepted — fal's default env name.)

Then run:

```bash
python ui/app.py
```

This launches a local Gradio app (usually http://127.0.0.1:7860).

### A1 — Segment
Upload an image, type a short phrase like "the hat", hit Segment.
**Backend → auto** tries SAM 3 first and falls back to YOLO-World + SAM 2.

### A2 — Inpaint
On the **Inpaint** tab: segment a region, pick an instance, type an edit prompt
(e.g. "a tall blue wizard hat"), choose backend:

| Backend | Behavior |
|---------|----------|
| `auto` | SDXL inpaint → quality gate → FLUX Kontext fill on API/gate fail |
| `sdxl` | `fal-ai/fast-sdxl/inpainting` only (still shows score card) |
| `flux_kontext` | `fal-ai/flux-pro/v1/fill` (masked) or `fal-ai/flux-kontext/dev` (instruction) |
| `animeadapter` | Stub — not available until weights ship |

**Quality gate** (default on): outside-mask fidelity + inside-mask change +
Moondream2 prompt adherence on the cropped edit region. Scores are logged to
`data/logs/inpaint_log.jsonl`.

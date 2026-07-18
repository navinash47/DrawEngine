# The Playground — Operation Shustrutha

Composable image-editing primitives, built backend-function-first so
ComicAgentEngine can call them directly later. The Gradio UI is a thin
layer for you (and beta testers) to drive them by eye.

## Status
- ✅ **A1 — Text-grounded segmentation** (SAM 3 via Roboflow's hosted API)
- ⬜ A2 — Masked inpainting
- ⬜ A3 — Identity conditioning (IP-Adapter / LoRA)
- ⬜ A4 — ControlNet stacking

## Setup (on your Mac — no local GPU needed for A1)

```bash
cd playground
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Get a free Roboflow API key at https://app.roboflow.com (Settings → API Keys),
put it in a repo-root `.env` file:

```bash
# DrawEngine/.env
ROBOFLOW_API_KEY=your_key_here
```

Then run:

```bash
python ui/app.py
```

This launches a local Gradio app (usually http://127.0.0.1:7860). Upload an
image, type a short phrase like "the hat" or "the girl in the red cloak",
and hit Segment.
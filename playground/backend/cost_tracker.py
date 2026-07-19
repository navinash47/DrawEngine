"""
cost_tracker.py — Estimated API spend logging for the playground.

Estimates (not fal invoices). Rates are easy to tweak in RATE_TABLE / ROBOFLOW_CALL_USD.
Persists to data/logs/api_cost_log.jsonl.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

_PLAYGROUND_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = _PLAYGROUND_ROOT / "data" / "logs"
COST_LOG = LOGS_DIR / "api_cost_log.jsonl"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Estimated USD rates (documented; adjust as fal pricing changes)
RATE_TABLE: dict[str, dict[str, float]] = {
    # megapixel-billed
    "fal-ai/flux-pro/v1/fill": {"usd_per_mp": 0.05},
    "fal-ai/flux-general": {"usd_per_mp": 0.075},
    # flat per successful call (compute-seconds model; typical inpaint)
    "fal-ai/fast-sdxl/inpainting": {"usd_per_call": 0.0025},
    "fal-ai/fast-sdxl": {"usd_per_call": 0.0025},
    # per 1000 characters of prompt + output
    "fal-ai/moondream2/visual-query": {"usd_per_1k_chars": 0.01},
    # flat per image
    "fal-ai/flux-kontext/dev": {"usd_per_image": 0.04},
}

ROBOFLOW_CALL_USD = 0.01

_lock = threading.Lock()
_session_usd = 0.0
_session_calls = 0


def reset_session() -> None:
    global _session_usd, _session_calls
    with _lock:
        _session_usd = 0.0
        _session_calls = 0


def session_totals() -> dict[str, float | int]:
    with _lock:
        return {"session_usd": _session_usd, "session_calls": _session_calls}


def _ceil_megapixels(width: int | None, height: int | None) -> float:
    if not width or not height:
        return 1.0
    return float(max(1, math.ceil((width * height) / 1_000_000)))


def estimate_fal_usd(
    model_id: str,
    *,
    arguments: dict | None = None,
    response: dict | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return (estimated_usd, units_meta)."""
    arguments = arguments or {}
    response = response or {}
    rates = RATE_TABLE.get(model_id, {"usd_per_call": 0.01})
    units: dict[str, Any] = {"model": model_id}

    if "usd_per_mp" in rates:
        w = h = None
        images = response.get("images")
        if isinstance(images, list) and images:
            first = images[0] if isinstance(images[0], dict) else {}
            w = first.get("width")
            h = first.get("height")
        mp = _ceil_megapixels(w, h)
        units.update({"billing": "megapixel", "megapixels": mp, "width": w, "height": h})
        return rates["usd_per_mp"] * mp, units

    if "usd_per_1k_chars" in rates:
        prompt = str(arguments.get("prompt") or "")
        out = str(
            response.get("output")
            or response.get("text")
            or response.get("answer")
            or ""
        )
        chars = len(prompt) + len(out)
        units.update({"billing": "chars", "chars": chars})
        return rates["usd_per_1k_chars"] * (chars / 1000.0), units

    if "usd_per_image" in rates:
        n = int(arguments.get("num_images") or 1)
        units.update({"billing": "image", "images": n})
        return rates["usd_per_image"] * n, units

    units.update({"billing": "call", "calls": 1})
    return float(rates.get("usd_per_call", 0.01)), units


def log_call(
    *,
    provider: str,
    model: str,
    operation: str,
    estimated_usd: float,
    units: dict[str, Any] | None = None,
    request_id: str | None = None,
    related_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one cost row and update session totals. Returns the row."""
    global _session_usd, _session_calls
    row: dict[str, Any] = {
        "timestamp": time.time(),
        "provider": provider,
        "model": model,
        "operation": operation,
        "units": units or {},
        "estimated_usd": round(float(estimated_usd), 6),
        "request_id": request_id,
        "related_id": related_id,
    }
    if extra:
        row["extra"] = extra

    with _lock:
        _session_usd += float(estimated_usd)
        _session_calls += 1
        with open(COST_LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
    return row


def log_fal_call(
    model_id: str,
    arguments: dict,
    response: dict,
    *,
    request_id: str | None = None,
    related_id: str | None = None,
    operation: str | None = None,
) -> dict[str, Any]:
    usd, units = estimate_fal_usd(model_id, arguments=arguments, response=response)
    op = operation or model_id.rsplit("/", 1)[-1]
    return log_call(
        provider="fal",
        model=model_id,
        operation=op,
        estimated_usd=usd,
        units=units,
        request_id=request_id,
        related_id=related_id,
    )


def log_roboflow_call(
    *,
    model: str,
    operation: str = "segment",
    request_id: str | None = None,
    related_id: str | None = None,
    estimated_usd: float | None = None,
) -> dict[str, Any]:
    usd = ROBOFLOW_CALL_USD if estimated_usd is None else estimated_usd
    return log_call(
        provider="roboflow",
        model=model,
        operation=operation,
        estimated_usd=usd,
        units={"billing": "call", "calls": 1},
        request_id=request_id,
        related_id=related_id,
    )


def _read_all() -> list[dict[str, Any]]:
    if not COST_LOG.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in COST_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize(recent_n: int = 50) -> dict[str, Any]:
    rows = _read_all()
    total = sum(float(r.get("estimated_usd") or 0) for r in rows)
    by_model: dict[str, float] = {}
    by_provider: dict[str, float] = {}
    for r in rows:
        m = str(r.get("model") or "unknown")
        p = str(r.get("provider") or "unknown")
        usd = float(r.get("estimated_usd") or 0)
        by_model[m] = by_model.get(m, 0.0) + usd
        by_provider[p] = by_provider.get(p, 0.0) + usd
    sess = session_totals()
    recent = rows[-recent_n:][::-1]  # newest first
    return {
        "total_usd": round(total, 6),
        "total_calls": len(rows),
        "session_usd": sess["session_usd"],
        "session_calls": sess["session_calls"],
        "by_model": {k: round(v, 6) for k, v in sorted(by_model.items(), key=lambda x: -x[1])},
        "by_provider": {k: round(v, 6) for k, v in sorted(by_provider.items(), key=lambda x: -x[1])},
        "recent": recent,
        "log_path": str(COST_LOG),
    }


def format_meter_markdown(summary: dict[str, Any] | None = None) -> str:
    s = summary or summarize()
    lines = [
        "### Expense meter *(estimated USD — not fal invoices)*",
        f"**Session:** `${s['session_usd']:.4f}` ({s['session_calls']} calls) · "
        f"**All-time:** `${s['total_usd']:.4f}` ({s['total_calls']} calls)",
        "",
        "#### By model",
    ]
    if s["by_model"]:
        for model, usd in s["by_model"].items():
            lines.append(f"- `{model}`: `${usd:.4f}`")
    else:
        lines.append("_No API calls logged yet._")
    lines.append("")
    lines.append(f"Log file: `{s['log_path']}`")
    return "\n".join(lines)


def recent_rows_for_table(recent_n: int = 40) -> list[list[Any]]:
    """Rows for a Gradio Dataframe: [time, provider, model, op, usd, request_id]."""
    from datetime import datetime

    s = summarize(recent_n=recent_n)
    out: list[list[Any]] = []
    for r in s["recent"]:
        ts = r.get("timestamp")
        try:
            tstr = datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError, OSError):
            tstr = str(ts)
        out.append(
            [
                tstr,
                r.get("provider"),
                r.get("model"),
                r.get("operation"),
                f"{float(r.get('estimated_usd') or 0):.4f}",
                r.get("request_id") or "",
            ]
        )
    return out

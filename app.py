from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
import time

import gradio as gr
import pandas as pd

from src.inference import GenreInferenceService


ROOT_DIR = Path(__file__).resolve().parent
EMPTY_TABLE = pd.DataFrame(columns=["genre", "probability"])
APP_CSS = """
:root {
    --bg-0: #eef2f7;
    --bg-1: #f7fafc;
    --surface-1: #ffffff;
    --surface-2: #f8fafc;
    --ink-0: #0f172a;
    --ink-1: #334155;
    --line: #cbd5e1;
    --accent: #0f766e;
    --accent-soft: rgba(15, 118, 110, 0.12);
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-0: #0b1220;
        --bg-1: #101a2e;
        --surface-1: #111c31;
        --surface-2: #16233d;
        --ink-0: #e2e8f0;
        --ink-1: #cbd5e1;
        --line: #24324d;
        --accent: #2dd4bf;
        --accent-soft: rgba(45, 212, 191, 0.16);
    }
}

[data-theme="dark"] {
    --bg-0: #0b1220;
    --bg-1: #101a2e;
    --surface-1: #111c31;
    --surface-2: #16233d;
    --ink-0: #e2e8f0;
    --ink-1: #cbd5e1;
    --line: #24324d;
    --accent: #2dd4bf;
    --accent-soft: rgba(45, 212, 191, 0.16);
}

.gradio-container {
    background:
        radial-gradient(circle at 14% 18%, var(--accent-soft), transparent 36%),
        radial-gradient(circle at 88% 4%, rgba(59, 130, 246, 0.16), transparent 34%),
        linear-gradient(145deg, var(--bg-1) 0%, var(--bg-0) 100%);
    color: var(--ink-0);
}

.app-shell {
    max-width: 1120px;
    margin: 0 auto;
    padding: 22px 8px 30px 8px;
}

.hero-card {
    background: var(--surface-1);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 16px 30px rgba(2, 6, 23, 0.06);
}

.hero-title {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.1px;
    color: var(--ink-0);
}

.hero-subtitle {
    margin: 10px 0 0 0;
    color: var(--ink-1);
    line-height: 1.5;
}

.panel {
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
}

.gr-block,
.gr-box,
.gr-form,
.gr-group,
.gr-dataframe,
.gr-code,
.gr-textbox,
.gr-audio {
    background: var(--surface-1) !important;
    border-color: var(--line) !important;
    color: var(--ink-0) !important;
}

.gr-markdown,
.gr-markdown p,
.gr-markdown li,
.gr-label {
    color: var(--ink-1) !important;
}

.gr-dataframe table,
.gr-dataframe th,
.gr-dataframe td {
    color: var(--ink-0) !important;
    background: var(--surface-1) !important;
}

.gr-button-primary {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #052e2b !important;
}
"""


@lru_cache(maxsize=1)
def get_service() -> GenreInferenceService:
    return GenreInferenceService(root_dir=ROOT_DIR)


def _safe_metadata(error: Exception | None = None, meta: dict | None = None) -> str:
    if error is not None:
        payload = {
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    payload = {"status": "ok", **(meta or {})}
    return json.dumps(payload, indent=2, sort_keys=True)


def _normalize_audio_path(audio_input) -> str:
    if audio_input is None:
        return ""

    if isinstance(audio_input, str):
        return audio_input

    if isinstance(audio_input, list):
        if not audio_input:
            return ""
        first = audio_input[0]
        return str(first) if first is not None else ""

    if hasattr(audio_input, "name"):
        return str(audio_input.name)

    return str(audio_input)


def _confidence_summary(table: pd.DataFrame) -> str:
    if table.empty or "probability" not in table.columns:
        return "Confidence: unavailable"

    probs = table["probability"].astype(float).tolist()
    top1 = probs[0]
    top2 = probs[1] if len(probs) > 1 else 0.0
    margin = top1 - top2

    if top1 >= 0.75 and margin >= 0.30:
        level = "High"
    elif top1 >= 0.50 and margin >= 0.15:
        level = "Medium"
    else:
        level = "Low"

    return (
        f"Confidence: {level} | "
        f"Top-1 probability: {top1:.3f} | "
        f"Margin vs Top-2: {margin:.3f}"
    )


def classify_audio(audio_path, top_k: int, tta_passes: int):
    audio_path = _normalize_audio_path(audio_path)
    if not audio_path:
        return (
            "Please upload an audio file.",
            "Confidence: unavailable",
            EMPTY_TABLE.copy(),
            _safe_metadata(),
        )

    try:
        top_k = max(1, min(int(top_k), 10))
        tta_passes = max(1, min(int(tta_passes), 32))
        start = time.perf_counter()
        service = get_service()
        pred, top, meta = service.predict(audio_path, top_k=top_k, tta_passes=tta_passes)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

        table = pd.DataFrame(top)
        if "probability" in table.columns:
            table["probability"] = table["probability"].map(lambda x: round(float(x), 6))

        meta["latency_ms"] = latency_ms
        meta["requested_top_k"] = top_k
        meta["requested_tta_passes"] = tta_passes

        label = f"Predicted genre: {pred}"
        return label, _confidence_summary(table), table, _safe_metadata(meta=meta)
    except Exception as exc:
        return (
            "Inference failed.",
            "Confidence: unavailable",
            EMPTY_TABLE.copy(),
            _safe_metadata(error=exc),
        )


def build_app() -> gr.Blocks:
    description = (
        "Upload an audio file and receive calibrated top-k genre probabilities from the production "
        "ResNet50 checkpoint. Configure TTA (10-15) for stronger and more stable predictions."
    )

    with gr.Blocks(title="Music Genre Classifier", css=APP_CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <section class='hero-card'>
                  <h1 class='hero-title'>Music Genre Classifier</h1>
                  <p class='hero-subtitle'>
                    Production-grade audio inference for genre recognition.
                    The model evaluates mel-spectrogram chunks and aggregates
                    probabilities for stable predictions across multiple TTA passes.
                  </p>
                </section>
                """
            )
            gr.Markdown(description)

            with gr.Row():
                with gr.Column(elem_classes=["panel"], scale=7):
                    audio_in = gr.Audio(
                        label="Upload audio file",
                        sources=["upload"],
                        type="filepath",
                    )
                with gr.Column(elem_classes=["panel"], scale=5):
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=5,
                        label="Top-k predictions",
                    )
                    tta_passes = gr.Slider(
                        minimum=10,
                        maximum=15,
                        step=1,
                        value=10,
                        label="TTA passes",
                        info="Higher values increase robustness but take longer.",
                    )
                    with gr.Row():
                        submit = gr.Button("Run inference", variant="primary")
                        clear = gr.Button("Clear", variant="secondary")

            pred_out = gr.Textbox(label="Predicted genre")
            confidence_out = gr.Textbox(label="Confidence summary")
            probs_out = gr.Dataframe(
                headers=["genre", "probability"],
                datatype=["str", "number"],
                interactive=False,
                label="Ranked probabilities",
            )
            meta_out = gr.Code(label="Inference metadata", language="json")

            submit.click(
                fn=classify_audio,
                inputs=[audio_in, top_k, tta_passes],
                outputs=[pred_out, confidence_out, probs_out, meta_out],
                api_name=False,
                show_api=False,
            )

            clear.click(
                fn=lambda: (
                    None,
                    5,
                    10,
                    "",
                    "Confidence: unavailable",
                    EMPTY_TABLE.copy(),
                    _safe_metadata(),
                ),
                inputs=None,
                outputs=[audio_in, top_k, tta_passes, pred_out, confidence_out, probs_out, meta_out],
                api_name=False,
                show_api=False,
            )

    return demo


demo = build_app()
# Prevent runtime schema introspection regressions in Gradio 4.44.x startup.
demo.api_info = {"named_endpoints": {}, "unnamed_endpoints": {}}
demo.all_app_info = demo.api_info


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_api=False,
        quiet=True,
    )

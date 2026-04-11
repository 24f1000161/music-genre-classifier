from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd

from src.inference import GenreInferenceService


ROOT_DIR = Path(__file__).resolve().parent
EMPTY_TABLE = pd.DataFrame(columns=["genre", "probability"])
APP_CSS = """
:root {
    --surface-0: #f4f6f8;
    --surface-1: #ffffff;
    --ink-0: #111827;
    --ink-1: #4b5563;
    --line: #d1d5db;
    --accent: #0f766e;
}

.gradio-container {
    background:
        radial-gradient(circle at 12% 18%, rgba(15, 118, 110, 0.14), transparent 36%),
        radial-gradient(circle at 88% 2%, rgba(59, 130, 246, 0.12), transparent 34%),
        linear-gradient(140deg, #f8fafc 0%, #edf2f7 100%);
    color: var(--ink-0);
}

.app-shell {
    max-width: 1080px;
    margin: 0 auto;
    padding: 20px 6px 28px 6px;
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
    letter-spacing: 0.2px;
}

.hero-subtitle {
    margin: 10px 0 0 0;
    color: var(--ink-1);
    line-height: 1.5;
}

.panel {
    background: var(--surface-1);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
}

.gr-button-primary {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
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


def classify_audio(audio_path: str, top_k: int):
    if not audio_path:
        return "Please upload an audio file.", EMPTY_TABLE.copy(), _safe_metadata()

    try:
        top_k = max(1, min(int(top_k), 10))
        service = get_service()
        pred, top, meta = service.predict(audio_path, top_k=top_k)
        table = pd.DataFrame(top)
        if "probability" in table.columns:
            table["probability"] = table["probability"].map(lambda x: round(float(x), 6))
        label = f"Predicted genre: {pred}"
        return label, table, _safe_metadata(meta=meta)
    except Exception as exc:
        return "Inference failed.", EMPTY_TABLE.copy(), _safe_metadata(error=exc)


def build_app() -> gr.Blocks:
    description = "Upload an audio file and receive calibrated top-k genre probabilities from the production ResNet50 checkpoint."

    with gr.Blocks(title="Music Genre Classifier", css=APP_CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <section class='hero-card'>
                  <h1 class='hero-title'>Music Genre Classifier</h1>
                  <p class='hero-subtitle'>
                    Production-grade audio inference for genre recognition.
                    The model evaluates mel-spectrogram chunks and aggregates
                    probabilities for stable predictions.
                  </p>
                </section>
                """
            )
            gr.Markdown(description)

            with gr.Row():
                with gr.Column(elem_classes=["panel"], scale=7):
                    audio_in = gr.Audio(type="filepath", label="Input audio")
                with gr.Column(elem_classes=["panel"], scale=5):
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=5,
                        label="Top-k predictions",
                    )
                    with gr.Row():
                        submit = gr.Button("Run inference", variant="primary")
                        clear = gr.Button("Clear", variant="secondary")

            pred_out = gr.Textbox(label="Predicted genre")
            probs_out = gr.Dataframe(
                headers=["genre", "probability"],
                datatype=["str", "number"],
                interactive=False,
                label="Ranked probabilities",
            )
            meta_out = gr.Code(label="Inference metadata", language="json")

            submit.click(
                fn=classify_audio,
                inputs=[audio_in, top_k],
                outputs=[pred_out, probs_out, meta_out],
                api_name=False,
                show_api=False,
            )

            clear.click(
                fn=lambda: (None, 5, "", EMPTY_TABLE.copy(), _safe_metadata()),
                inputs=None,
                outputs=[audio_in, top_k, pred_out, probs_out, meta_out],
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

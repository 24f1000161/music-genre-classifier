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
        return "No audio provided.", EMPTY_TABLE.copy(), _safe_metadata()

    try:
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
    description = (
        "Upload an audio file to classify its genre using the best ResNet50 sprint checkpoint "
        "(W&B run 7w7bs387). The model uses log-mel spectrogram preprocessing and multi-chunk "
        "probability averaging for robust predictions. <br><br>"
        "🏆 **Performance:** Achieved **0.89** validation accuracy and **0.93** testing accuracy "
        "in the Kaggle competition!"
    )

    with gr.Blocks(title="Music Genre Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown("<h1 style='text-align: center;'>🎵 Music Genre Classifier 🎧</h1>")
        gr.Markdown(f"<p style='text-align: center; font-size: 1.1em;'>{description}</p>")

        with gr.Row():
            audio_in = gr.Audio(type="filepath", label="Input Audio")
            with gr.Column():
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    label="Top-K predictions",
                )
                submit = gr.Button("Classify", variant="primary", size="lg")

        with gr.Row():
            pred_out = gr.Textbox(label="Prediction", scale=1)
            probs_out = gr.Dataframe(headers=["genre", "probability"], label="Top predictions", scale=2)
            meta_out = gr.Textbox(label="Inference metadata", lines=10, scale=1)

        submit.click(
            fn=classify_audio,
            inputs=[audio_in, top_k],
            outputs=[pred_out, probs_out, meta_out],
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
        show_api=False,
        quiet=True,
    )

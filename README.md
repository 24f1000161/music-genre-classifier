---
title: Music Genre Classifier
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
---

# Music Genre Classifier (ResNet50 Sprint)

This Space deploys the best checkpoint from W&B run 7w7bs387:
- File: resnet50_1hour_best.pth
- Notebook lineage: resnet50_1hour_sprint
- Reported best validation macro-F1: approximately 0.89

Runtime model loading behavior:
- If `checkpoints/manifest.json` exists, the app reconstructs the model from local parts.
- Otherwise, it downloads `resnet50_1hour_best.pth` from W&B on first startup.

Environment variables:
- `WANDB_API_KEY`: optional if the run/file is private.
- `WANDB_RUN_PATH`: defaults to `24f1000161-dl-genai-project/Messy-Mashup-Cutoff/7w7bs387`.
- `WANDB_MODEL_FILE`: defaults to `resnet50_1hour_best.pth`.
- `WANDB_MODEL_URL`: optional direct file URL fallback.

UV project workflow:
- `pyproject.toml` and `uv.lock` are included for uv-based dependency management.
- Create/sync env locally with `uv sync --python 3.10`.
- Run app locally with `uv run python app.py`.
- To refresh lock with newest allowed versions, run `uv lock --upgrade`.

Note: Gradio SDK Spaces still install `requirements.txt` via pip at build time.
Keep `requirements.txt` aligned with `pyproject.toml` when changing dependencies.

Compatibility pinning:
- Keep `fastapi==0.112.2` and `starlette==0.38.6` with Gradio 4.44.x.
- Newer Starlette 1.x can break Gradio 4.44 template rendering with `TypeError: unhashable type: 'dict'`.

To create/push a Hugging Face Space repo from this folder:
1. Export a write token: `export HF_TOKEN=...`
2. Run: `python3 tools/push_to_hf_space.py`

The app performs robust audio genre classification by:
- resampling audio to the training sample rate
- creating log-mel spectrogram images that match training preprocessing
- averaging probabilities across multiple audio chunks


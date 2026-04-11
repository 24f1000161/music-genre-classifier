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

# Music Genre Classifier

Production-grade music genre classification web app powered by a ResNet50 checkpoint.

## Live Deployment

- Hugging Face Space: https://huggingface.co/spaces/iamdivyam/GENRES-OF-MUSIC

## Overview

This application predicts genre labels from uploaded audio using a robust inference pipeline:

- log-mel spectrogram image conversion
- chunked evaluation over long audio
- multi-pass test-time augmentation (TTA) ensembling

The deployed model originates from the best checkpoint of W&B run `7w7bs387`.

## Product Features

- Dark-mode-safe and light-mode-safe interface
- Upload-and-predict workflow with ranked top-k classes
- Configurable TTA slider directly in the UI (10 to 15 passes)
- Confidence summary for each prediction
- Probability profile chart for quick interpretation
- Detailed inference metadata (latency, TTA settings, device)

## Reliability Tooling

This project introduces an **Audio Reliability Layer** for decoding stability across environments:

- primary decoder: system `ffmpeg`
- fallback decoder: explicit torchaudio legacy backends (`soundfile`, `ffmpeg`, `sox_io`)
- avoids default torchaudio loader paths that can fail with TorchCodec-only setups

Result: more stable behavior in containerized deployments (including Hugging Face Spaces).

## Model Loading Strategy

- If `checkpoints/manifest.json` exists, the model can be assembled from local parts.
- Otherwise, inference uses `models/resnet50_1hour_best.pth`.
- Optional override via `LOCAL_CHECKPOINT_PATH`.
- No runtime download from W&B is required.

## Inference Quality Configuration

Default TTA is intentionally stronger for public inference:

- baseline: **10 passes**
- optional higher setting: **15 passes**

Use environment variable override:

```bash
export INFERENCE_TTA_PASSES=15
```

At runtime, users can also set TTA from the app UI using the **TTA passes** slider.

## Local Development

### Option A: pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Option B: uv

```bash
uv sync --python 3.10
uv run python app.py
```

## Environment Variables

- `LOCAL_CHECKPOINT_PATH`: optional path to a local `.pth` checkpoint.
- `INFERENCE_TTA_PASSES`: optional override for TTA pass count (for example `10` or `15`).

## Compatibility Notes

- `gradio==4.44.1`
- `fastapi==0.112.2`
- `starlette==0.38.6`

These pins are intentional to avoid known runtime incompatibilities with Gradio 4.44.x.

## Deployment Notes

The Space uses Gradio SDK with `requirements.txt` installation during build.
Keep `requirements.txt` and `pyproject.toml` aligned whenever dependencies are changed.


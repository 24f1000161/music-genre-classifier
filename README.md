---
title: Music Genre Classifier
emoji: 🎵
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
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

The app performs robust audio genre classification by:
- resampling audio to the training sample rate
- creating log-mel spectrogram images that match training preprocessing
- averaging probabilities across multiple audio chunks


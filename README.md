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

The app performs robust audio genre classification by:
- resampling audio to the training sample rate
- creating log-mel spectrogram images that match training preprocessing
- averaging probabilities across multiple audio chunks


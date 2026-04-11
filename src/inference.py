from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .audio import build_mel_frontend, create_eval_chunks, load_waveform_mono, wave_to_image
from .checkpoint import ensure_checkpoint
from .config import DEFAULT_GENRES, InferenceConfig
from .model import load_model_from_checkpoint


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class GenreInferenceService:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.checkpoints_dir = root_dir / "checkpoints"
        self.models_dir = root_dir / "models"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = ensure_checkpoint(self.checkpoints_dir, self.models_dir)
        checkpoint = _safe_torch_load(checkpoint_path)

        ckpt_cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint, dict) else {}
        if not isinstance(ckpt_cfg, dict):
            ckpt_cfg = {}
        self.cfg = self._build_cfg(ckpt_cfg)

        genres = checkpoint.get("genres") if isinstance(checkpoint, dict) else None
        self.genres = genres if isinstance(genres, list) and len(genres) > 0 else DEFAULT_GENRES

        self.model = load_model_from_checkpoint(
            checkpoint=checkpoint,
            num_classes=len(self.genres),
            device=self.device,
        )

        self.mel_transform, self.db_transform = build_mel_frontend(
            sample_rate=self.cfg.sr,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            device=self.device,
        )

    @staticmethod
    def _build_cfg(raw_cfg: Dict) -> InferenceConfig:
        merged = asdict(InferenceConfig())
        for key in merged:
            if key in raw_cfg and raw_cfg[key] is not None:
                merged[key] = raw_cfg[key]

        env_tta = os.getenv("INFERENCE_TTA_PASSES", "").strip()
        if env_tta:
            try:
                merged["tta_passes"] = max(1, int(env_tta))
            except ValueError:
                pass
        else:
            # Keep a stronger baseline even when older checkpoints contain lower TTA values.
            merged["tta_passes"] = max(10, int(merged.get("tta_passes", 10)))

        return InferenceConfig(**merged)

    @torch.no_grad()
    def _predict_pass(self, chunks: List[torch.Tensor]) -> np.ndarray:
        probs_all: List[np.ndarray] = []
        bs = max(1, self.cfg.val_bs)

        for idx in range(0, len(chunks), bs):
            batch = torch.stack(chunks[idx : idx + bs]).to(self.device).float()
            images = wave_to_image(batch, self.mel_transform, self.db_transform)
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)

        pass_probs = np.concatenate(probs_all, axis=0)
        return pass_probs.mean(axis=0)

    def _build_tta_variants(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        variants = [waveform]
        tta = max(1, int(self.cfg.tta_passes))
        if tta <= 1:
            return variants

        shift = max(1, int(self.cfg.sr * 0.25))
        for i in range(1, tta):
            variants.append(torch.roll(waveform, shifts=i * shift, dims=1))
        return variants

    def predict(self, audio_path: str, top_k: int = 5) -> Tuple[str, List[Dict[str, float]], Dict]:
        waveform = load_waveform_mono(audio_path, target_sr=self.cfg.sr)
        target_samples = int(self.cfg.sr * self.cfg.duration)

        tta_probs: List[np.ndarray] = []
        num_chunks = 0
        for variant in self._build_tta_variants(waveform):
            chunks = create_eval_chunks(variant, target_samples=target_samples)
            num_chunks = max(num_chunks, len(chunks))
            tta_probs.append(self._predict_pass(chunks))

        probs = np.stack(tta_probs, axis=0).mean(axis=0)
        probs = probs / np.clip(probs.sum(), 1e-8, None)

        pred_idx = int(np.argmax(probs))
        pred_genre = self.genres[pred_idx]

        k = max(1, min(int(top_k), len(self.genres)))
        top_indices = np.argsort(-probs)[:k]
        top = [
            {"genre": self.genres[int(i)], "probability": float(probs[int(i)])}
            for i in top_indices
        ]

        meta = {
            "sample_rate": self.cfg.sr,
            "duration_sec": self.cfg.duration,
            "tta_passes": self.cfg.tta_passes,
            "num_chunks": num_chunks,
            "device": str(self.device),
            "checkpoint_source": "local",
        }
        return pred_genre, top, meta

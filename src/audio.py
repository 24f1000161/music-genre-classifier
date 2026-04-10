from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


def load_waveform_mono(path: str, target_sr: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.dim() != 2:
        raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def create_eval_chunks(waveform: torch.Tensor, target_samples: int) -> List[torch.Tensor]:
    total = waveform.shape[1]
    if total <= target_samples:
        padded = torch.zeros((1, target_samples), dtype=waveform.dtype)
        padded[:, :total] = waveform
        return [padded.squeeze(0)]

    stride = max(1, target_samples // 2)
    chunks: List[torch.Tensor] = []
    start = 0
    while start < total:
        end = start + target_samples
        seg = waveform[:, start:end]
        if seg.shape[1] < target_samples:
            padded = torch.zeros((1, target_samples), dtype=waveform.dtype)
            padded[:, : seg.shape[1]] = seg
            seg = padded
        chunks.append(seg.squeeze(0))
        if end >= total:
            break
        start += stride

    return chunks


def build_mel_frontend(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: int,
    fmax: int,
    device: torch.device,
):
    mel = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
    ).to(device)
    db = T.AmplitudeToDB(top_db=80).to(device)
    return mel, db


def wave_to_image(
    waves: torch.Tensor,
    mel_transform: T.MelSpectrogram,
    db_transform: T.AmplitudeToDB,
) -> torch.Tensor:
    mel = mel_transform(waves)
    mel = db_transform(mel)
    mel = mel.unsqueeze(1)
    mel = F.interpolate(mel, size=(224, 224), mode="bilinear", align_corners=False)
    mmin = mel.amin(dim=(1, 2, 3), keepdim=True)
    mmax = mel.amax(dim=(1, 2, 3), keepdim=True)
    mel = (mel - mmin) / (mmax - mmin + 1e-6)
    return mel

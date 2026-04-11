from __future__ import annotations

import subprocess
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


def _to_mono_resampled(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")

    waveform = waveform.float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def _load_with_ffmpeg(path: str, target_sr: int) -> tuple[torch.Tensor, int]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or "ffmpeg audio decode failed")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("ffmpeg returned empty audio stream")

    waveform = torch.from_numpy(audio).unsqueeze(0)
    return waveform, target_sr


def _load_with_torchaudio_explicit(path: str) -> tuple[torch.Tensor, int]:
    # Use explicit backends only; avoid torchaudio default loader path.
    backend_candidates = ["soundfile", "ffmpeg", "sox_io"]
    errors: List[str] = []

    for backend in backend_candidates:
        try:
            try:
                waveform, sr = torchaudio.load(path, backend=backend)
            except TypeError:
                # Compatibility path for older torchaudio versions.
                torchaudio.set_audio_backend(backend)
                waveform, sr = torchaudio.load(path)
            return waveform, sr
        except Exception as exc:
            errors.append(f"{backend}: {type(exc).__name__}: {exc}")

    raise RuntimeError("; ".join(errors))


def load_waveform_mono(path: str, target_sr: int) -> torch.Tensor:
    errors: List[str] = []

    try:
        waveform, sr = _load_with_ffmpeg(path, target_sr=target_sr)
        return _to_mono_resampled(waveform, sr=sr, target_sr=target_sr)
    except Exception as exc:
        errors.append(f"ffmpeg: {type(exc).__name__}: {exc}")

    try:
        waveform, sr = _load_with_torchaudio_explicit(path)
        return _to_mono_resampled(waveform, sr=sr, target_sr=target_sr)
    except Exception as exc:
        errors.append(f"torchaudio: {type(exc).__name__}: {exc}")

    detail = " | ".join(errors)
    raise RuntimeError(f"Failed to decode audio file '{path}'. {detail}")


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

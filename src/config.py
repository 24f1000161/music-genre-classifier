from __future__ import annotations

from dataclasses import dataclass
from typing import List


DEFAULT_GENRES: List[str] = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


@dataclass(frozen=True)
class InferenceConfig:
    sr: int = 22050
    duration: int = 8
    n_mels: int = 224
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 11025
    val_bs: int = 16
    tta_passes: int = 2


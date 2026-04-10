from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50


def build_resnet50_for_mel(num_classes: int) -> nn.Module:
    model = resnet50(weights=None)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint_state_dict(checkpoint: dict) -> dict:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def load_model_from_checkpoint(
    checkpoint: dict,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    model = build_resnet50_for_mel(num_classes=num_classes)
    state_dict = load_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

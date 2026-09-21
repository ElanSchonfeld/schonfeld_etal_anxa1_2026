"""Torch helpers (device selection) for optional GPU acceleration."""

from __future__ import annotations

import os
from typing import Optional


def resolve_torch_device(requested: Optional[str] = None):
    """Pick a torch.device from an explicit request or environment."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - torch optional
        raise ImportError("torch is required for device selection") from exc

    if requested:
        return torch.device(requested)

    env = os.getenv("MOE_TORCH_DEVICE", "").strip()
    if env:
        return torch.device(env)

    return torch.device("cpu")

"""Model I/O helpers for MoE v4 (stub)."""

from __future__ import annotations

from typing import Any

from .moe_types import ModelBundle


def save_model(bundle: ModelBundle, path: str) -> None:
    """Persist a ModelBundle to disk (placeholder)."""

    raise NotImplementedError("save_model is not implemented yet.")


def load_model(path: str) -> ModelBundle:
    """Load a ModelBundle from disk (placeholder)."""

    raise NotImplementedError("load_model is not implemented yet.")


def adapt_legacy_model(obj: Any) -> ModelBundle:
    raise NotImplementedError("adapt_legacy_model is not implemented yet.")

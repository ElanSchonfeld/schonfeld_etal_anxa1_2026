"""Lightweight progress helpers for v4 training loops."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Set, TypeVar

_T = TypeVar("_T")
_SEEN_DESCS: Set[str] = set()


def _progress_enabled(enabled: Optional[bool] = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    env = os.getenv("MOE_PROGRESS", "1").strip().lower()
    return env not in {"0", "false", "no", "off"}


def progress(
    iterable: Iterable[_T],
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    enabled: Optional[bool] = None,
):
    """Wrap an iterable with tqdm if available and enabled."""
    if not _progress_enabled(enabled):
        return iterable
    mode = os.getenv("MOE_PROGRESS_MODE", "collapse").strip().lower()
    if mode in {"none", "off"}:
        return iterable
    if mode == "collapse" and desc:
        if desc in _SEEN_DESCS:
            return iterable
        _SEEN_DESCS.add(desc)
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return iterable
    leave = mode not in {"collapse"}
    return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)

"""Typed structures for MoE v4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


@dataclass(frozen=True)
class Node:
    """Hierarchy node metadata."""

    path: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Expert:
    """Expert metadata placeholder."""

    name: str
    classes: List[str]
    payload: Any = None


@dataclass(frozen=True)
class Gate:
    """Gate (stacker) metadata placeholder."""

    name: str
    classes: List[str]
    payload: Any = None


class ModelBundle(TypedDict, total=False):
    """TypedDict wrapper for MoE model components."""

    experts: Dict[str, Expert]
    gates: Dict[str, Gate]
    hierarchy: Dict[str, Any]
    metadata: Dict[str, Any]

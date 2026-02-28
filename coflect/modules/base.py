"""Module abstraction for pluggable Coflect modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleSpec:
    """Static metadata for a Coflect module."""

    name: str
    description: str
    backends: tuple[str, ...]



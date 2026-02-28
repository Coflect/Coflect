"""Registry helpers for Coflect modules."""

from __future__ import annotations

from coflect.modules.base import ModuleSpec

_MODULES: dict[str, ModuleSpec] = {}


def register_module(spec: ModuleSpec) -> None:
    """Register a module specification by its canonical name.

    Example:
        >>> register_module(spec)
    """
    key = spec.name.strip().lower()
    if not key:
        raise ValueError("module name cannot be empty")
    _MODULES[key] = spec


def get_module(name: str) -> ModuleSpec:
    """Return module specification by name.

    Example:
        >>> hitl = get_module("hitl")
    """
    key = name.strip().lower()
    if key not in _MODULES:
        available = ", ".join(sorted(_MODULES)) or "<none>"
        raise KeyError(f"Unknown module '{name}'. Registered: {available}")
    return _MODULES[key]


def list_modules() -> tuple[str, ...]:
    """List registered module names.

    Example:
        >>> names = list_modules()
    """
    return tuple(sorted(_MODULES))

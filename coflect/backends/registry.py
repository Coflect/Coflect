"""Registry helpers for backend adapter factories."""

from __future__ import annotations

from collections.abc import Callable

from coflect.backends.base import BackendAdapter

BackendFactory = Callable[..., BackendAdapter]

_BACKEND_FACTORIES: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a backend adapter factory by name.

    Example:
        >>> register_backend("torch", factory)
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("backend name cannot be empty")
    _BACKEND_FACTORIES[key] = factory


def get_backend_factory(name: str) -> BackendFactory:
    """Return a previously registered backend factory.

    Example:
        >>> fn = get_backend_factory("torch")
    """
    key = name.strip().lower()
    if key not in _BACKEND_FACTORIES:
        available = ", ".join(sorted(_BACKEND_FACTORIES)) or "<none>"
        raise KeyError(f"Unknown backend '{name}'. Registered: {available}")
    return _BACKEND_FACTORIES[key]


def list_backends() -> tuple[str, ...]:
    """List registered backends.

    Example:
        >>> names = list_backends()
    """
    return tuple(sorted(_BACKEND_FACTORIES))

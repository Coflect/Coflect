"""JAX backend skeleton.

This is intentionally lightweight and dependency-free at import time.
Concrete implementation lands once JAX integration tests are in place.
"""

from __future__ import annotations

from typing import Any

from coflect.backends.base import BackendAdapter


class JaxAdapter(BackendAdapter):
    """Placeholder adapter for planned JAX backend support."""

    def forward(self, x: Any) -> Any:
        """Placeholder forward pass for future JAX backend.

        Example:
            >>> JaxAdapter().forward(None)
            Traceback (most recent call last):
            ...
            NotImplementedError
        """
        raise NotImplementedError("JaxAdapter is not implemented yet")

    def loss(self, logits: Any, y: Any) -> Any:
        """Placeholder loss function for future JAX backend.

        Example:
            >>> JaxAdapter().loss(None, None)
            Traceback (most recent call last):
            ...
            NotImplementedError
        """
        raise NotImplementedError("JaxAdapter is not implemented yet")

    def step(self, loss: Any) -> None:
        """Placeholder optimizer step for future JAX backend.

        Example:
            >>> JaxAdapter().step(None)
            Traceback (most recent call last):
            ...
            NotImplementedError
        """
        raise NotImplementedError("JaxAdapter is not implemented yet")

    def attribution(self, x_chw: Any, target_class: int, method: str = "livecam") -> bytes:
        """Placeholder attribution API for future JAX backend.

        Example:
            >>> JaxAdapter().attribution(None, target_class=0)
            Traceback (most recent call last):
            ...
            NotImplementedError
        """
        raise NotImplementedError("JaxAdapter is not implemented yet")

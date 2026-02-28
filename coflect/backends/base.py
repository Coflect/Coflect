"""Backend adapter interface for framework-agnostic training logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BackendAdapter(ABC):
    """Minimal backend contract inspired by thin adapter designs in major ML stacks."""

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Run model forward pass.

        Example:
            >>> logits = adapter.forward(batch_x)
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, logits: Any, y: Any) -> Any:
        """Compute primary supervised loss.

        Example:
            >>> loss = adapter.loss(logits, batch_y)
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, loss: Any) -> None:
        """Apply one optimizer step.

        Example:
            >>> adapter.step(loss)
        """
        raise NotImplementedError

    @abstractmethod
    def attribution(self, x_chw: Any, target_class: int, method: str = "livecam") -> bytes:
        """Return encoded overlay bytes for attribution visualization.

        Example:
            >>> png = adapter.attribution(x_chw, target_class=1, method="consensus")
        """
        raise NotImplementedError

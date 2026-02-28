"""PyTorch backend adapter implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from coflect.backends.base import BackendAdapter
from coflect.modules.hitl.xai_worker.livecam import make_overlay_png


class TorchAdapter(BackendAdapter):
    """Thin adapter around a torch model/optimizer/loss."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: optim.Optimizer | None,
        criterion: nn.Module | None,
    ):
        """Store model components used by the trainer hot path.

        Example:
            >>> adapter = TorchAdapter(model=m, optimizer=opt, criterion=loss_fn)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Example:
            >>> logits = adapter.forward(batch_x)
        """
        return self.model(x)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute scalar training loss from logits and labels.

        Example:
            >>> loss = adapter.loss(logits, batch_y)
        """
        if self.criterion is None:
            raise RuntimeError("TorchAdapter.criterion is not configured")
        return self.criterion(logits, y)

    def step(self, loss: torch.Tensor) -> None:
        """Apply one optimizer step.

        Example:
            >>> adapter.step(loss)
        """
        if self.optimizer is None:
            raise RuntimeError("TorchAdapter.optimizer is not configured")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    def attribution(self, x_chw: torch.Tensor, target_class: int, method: str = "livecam") -> bytes:
        """Return PNG bytes for a single attribution overlay.

        Example:
            >>> png = adapter.attribution(x_chw, target_class=1, method="consensus")
        """
        return make_overlay_png(self.model, x_chw, target_class=target_class, method=method)

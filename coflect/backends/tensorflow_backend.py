"""TensorFlow/Keras backend adapter implementation."""

from __future__ import annotations

from typing import Any

from coflect.backends.base import BackendAdapter


class TensorFlowAdapter(BackendAdapter):
    """Thin adapter around a tf.keras model/optimizer/loss."""

    def __init__(self, model: Any, optimizer: Any, loss_fn: Any | None = None):
        """Store TensorFlow model components used by trainer loops.

        Example:
            >>> adapter = TensorFlowAdapter(model=m, optimizer=opt, loss_fn=loss_fn)
        """
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("TensorFlow is required; install with `pip install coflect[tensorflow]`") from exc

        self.tf = tf
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def forward(self, x: Any) -> Any:
        """Run training forward pass.

        Example:
            >>> logits = adapter.forward(batch_x)
        """
        return self.model(x, training=True)

    def loss(self, logits: Any, y: Any) -> Any:
        """Compute loss tensor from logits and labels.

        Example:
            >>> loss = adapter.loss(logits, batch_y)
        """
        return self.loss_fn(y, logits)

    def step(self, loss: Any) -> None:
        """Reject direct stepping because trainer uses GradientTape.

        Example:
            >>> adapter.step(loss)
            Traceback (most recent call last):
            ...
            RuntimeError
        """
        raise RuntimeError("TensorFlowAdapter.step is not supported directly; use GradientTape in trainer")

    def attribution(self, x_chw: Any, target_class: int, method: str = "livecam") -> bytes:
        """Return PNG bytes for one TensorFlow attribution overlay.

        Example:
            >>> png = adapter.attribution(x_chw, target_class=2, method="consensus")
        """
        from coflect.modules.hitl.xai_worker.livecam_tf import make_overlay_png_tf_with_meta

        png, _ = make_overlay_png_tf_with_meta(
            self.model,
            x_chw,
            target_class=target_class,
            method=method,
        )
        return png

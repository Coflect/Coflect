"""Structured message payloads shared across trainer/worker/backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class XaiRequestPayload:
    """Minimal XAI request payload sent from trainer to backend queue."""

    step: int
    sample_idx: int
    target_class: int
    pred_class: int
    request_kind: str = "periodic"
    risk_score: float | None = None
    horizon_epochs: int | None = None
    backend: str = "torch"

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass payload to backend request dictionary.

        Example:
            >>> XaiRequestPayload(step=1, sample_idx=2, target_class=0, pred_class=1).to_dict()[\"step\"]
            1
        """
        return asdict(self)

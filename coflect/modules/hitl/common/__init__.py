"""Shared HITL message, parsing, notebook bridge, and wire utilities."""

from coflect.modules.hitl.common.notebook_bridge import (
    FeedbackUpdate,
    NotebookBridgeConfig,
    NotebookHITLBridge,
    roi_norm_to_pixels,
)

__all__ = [
    "NotebookBridgeConfig",
    "NotebookHITLBridge",
    "FeedbackUpdate",
    "roi_norm_to_pixels",
]

"""Backward-compatible wrapper for legacy Torch Grad-CAM worker module name."""

from __future__ import annotations

from coflect.modules.hitl.xai_worker.worker_torch_livecam import main

if __name__ == "__main__":
    main()


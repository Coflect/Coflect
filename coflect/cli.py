"""Command-line entrypoints for Coflect package scripts."""

from __future__ import annotations


def run_hitl_backend() -> None:
    """Start HITL backend API server.

    Example:
        >>> # CLI: coflect-hitl-backend --host 127.0.0.1 --port 8000
    """
    from coflect.modules.hitl.backend.app import main

    main()


def run_hitl_trainer_torch() -> None:
    """Run HITL trainer with Torch backend.

    Example:
        >>> # CLI: coflect-hitl-trainer-torch --server http://localhost:8000
    """
    from coflect.modules.hitl.trainer.train_torch import main

    main()


def run_hitl_xai_worker_torch() -> None:
    """Run HITL XAI worker with Torch backend.

    Example:
        >>> # CLI: coflect-hitl-xai-worker-torch --server http://localhost:8000
    """
    from coflect.modules.hitl.xai_worker.worker_torch_livecam import main

    main()


def run_hitl_trainer_tf() -> None:
    """Run HITL trainer with TensorFlow/Keras backend.

    Example:
        >>> # CLI: coflect-hitl-trainer-tf --server http://localhost:8000
    """
    from coflect.modules.hitl.trainer.train_tf import main

    main()


def run_hitl_xai_worker_tf() -> None:
    """Run HITL XAI worker with TensorFlow/Keras backend.

    Example:
        >>> # CLI: coflect-hitl-xai-worker-tf --server http://localhost:8000
    """
    from coflect.modules.hitl.xai_worker.worker_tf_livecam import main

    main()


def run_hitl_forecast_worker() -> None:
    """Run CPU-only HITL forecast worker.

    Example:
        >>> # CLI: coflect-hitl-forecast-worker --server http://localhost:8000 --backend torch
    """
    from coflect.modules.hitl.forecast.worker import main

    main()


def run_hitl_stack() -> None:
    """Run backend, trainer, forecast worker, and XAI worker together.

    Example:
        >>> # CLI: coflect-hitl-run --backend torch --dataset synthetic
    """
    from coflect.modules.hitl.launcher import main

    main()

"""Command-line entrypoints for Coflect package scripts."""

from __future__ import annotations


def run_hilt_backend() -> None:
    """Start HILT backend API server.

    Example:
        >>> # CLI: coflect-hilt-backend --host 127.0.0.1 --port 8000
    """
    from coflect.modules.hilt.backend.app import main

    main()


def run_hilt_ui() -> None:
    """Start HILT UI/backend server.

    This is an alias of `coflect-hilt-backend` for notebook workflows.

    Example:
        >>> # CLI: coflect-hilt-ui --host 127.0.0.1 --port 8000
    """
    from coflect.modules.hilt.backend.app import main

    main()


def run_hilt_trainer_torch() -> None:
    """Run HILT trainer with Torch backend.

    Example:
        >>> # CLI: coflect-hilt-trainer-torch --server http://localhost:8000
    """
    from coflect.modules.hilt.trainer.train_torch import main

    main()


def run_hilt_xai_worker_torch() -> None:
    """Run HILT XAI worker with Torch backend.

    Example:
        >>> # CLI: coflect-hilt-xai-worker-torch --server http://localhost:8000
    """
    from coflect.modules.hilt.xai_worker.worker_torch_livecam import main

    main()


def run_hilt_trainer_tf() -> None:
    """Run HILT trainer with TensorFlow/Keras backend.

    Example:
        >>> # CLI: coflect-hilt-trainer-tf --server http://localhost:8000
    """
    from coflect.modules.hilt.trainer.train_tf import main

    main()


def run_hilt_xai_worker_tf() -> None:
    """Run HILT XAI worker with TensorFlow/Keras backend.

    Example:
        >>> # CLI: coflect-hilt-xai-worker-tf --server http://localhost:8000
    """
    from coflect.modules.hilt.xai_worker.worker_tf_livecam import main

    main()


def run_hilt_forecast_worker() -> None:
    """Run CPU-only HILT forecast worker.

    Example:
        >>> # CLI: coflect-hilt-forecast-worker --server http://localhost:8000 --backend torch
    """
    from coflect.modules.hilt.forecast.worker import main

    main()


def run_hilt_stack() -> None:
    """Run backend, trainer, forecast worker, and XAI worker together.

    Example:
        >>> # CLI: coflect-hilt-run --backend torch --dataset synthetic
    """
    from coflect.modules.hilt.launcher import main

    main()

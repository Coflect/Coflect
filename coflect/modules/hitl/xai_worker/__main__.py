"""Module runner for `python -m coflect.modules.hitl.xai_worker`."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Route module execution to backend-specific XAI worker runner."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=["torch", "tensorflow"], default="torch")
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]

    if args.backend == "tensorflow":
        from coflect.modules.hitl.xai_worker.worker_tf_livecam import main as run_main
    else:
        from coflect.modules.hitl.xai_worker.worker_torch_livecam import main as run_main

    run_main()


if __name__ == "__main__":
    main()

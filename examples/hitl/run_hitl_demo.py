"""Utility script for HITL local demo commands.

By default this prints commands. Use `--launch` to run backend, trainer,
forecast worker, and XAI worker. The UI is served by backend at
`http://localhost:8000`.
"""

from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import time
from pathlib import Path

BACKEND_CMD = "coflect-hitl-backend --host 0.0.0.0 --port 8000"
TRAINER_CMDS = {
    "torch": (
        "coflect-hitl-trainer-torch --server http://localhost:8000 "
        "--steps 1000 --xai_every 100 --forecast_every 20"
    ),
    "tensorflow": (
        "coflect-hitl-trainer-tf --server http://localhost:8000 "
        "--steps 1000 --xai_every 100 --forecast_every 20"
    ),
}
FORECAST_CMDS = {
    "torch": "coflect-hitl-forecast-worker --server http://localhost:8000 --backend torch",
    "tensorflow": "coflect-hitl-forecast-worker --server http://localhost:8000 --backend tensorflow",
}
WORKER_CMDS = {
    "torch": "coflect-hitl-xai-worker-torch --server http://localhost:8000 --xai_method consensus",
    "tensorflow": "coflect-hitl-xai-worker-tf --server http://localhost:8000 --xai_method consensus",
}
UI_URL = "http://localhost:8000"


def _compose_torch_cmds(dataset: str, data_root: str, download_data: bool) -> tuple[str, str]:
    trainer = TRAINER_CMDS["torch"]
    worker = WORKER_CMDS["torch"]
    if dataset == "cifar10_catsdogs":
        trainer += f" --dataset {dataset} --data_root {shlex.quote(data_root)}"
        worker += f" --dataset {dataset} --data_root {shlex.quote(data_root)}"
        if download_data:
            trainer += " --download_data"
            worker += " --download_data"
    return trainer, worker


def _print_commands(backend: str, dataset: str, data_root: str, download_data: bool) -> tuple[str, str]:
    trainer_cmd = TRAINER_CMDS[backend]
    worker_cmd = WORKER_CMDS[backend]
    if backend == "torch":
        trainer_cmd, worker_cmd = _compose_torch_cmds(dataset=dataset, data_root=data_root, download_data=download_data)
    print("Run these in separate terminals:")
    print(f"1) {BACKEND_CMD}")
    print(f"2) {trainer_cmd}")
    print(f"3) {FORECAST_CMDS[backend]}")
    print(f"4) {worker_cmd}")
    print(f"5) Open browser: {UI_URL}")
    return trainer_cmd, worker_cmd


def _launch_process(cmd: str, cwd: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(shlex.split(cmd), cwd=str(cwd))


def main() -> None:
    parser = argparse.ArgumentParser(description="HITL demo command helper")
    parser.add_argument("--launch", action="store_true", help="Launch backend/trainer/forecast/xai workers")
    parser.add_argument("--duration", type=int, default=30, help="Seconds to keep launched processes alive")
    parser.add_argument("--backend", choices=["torch", "tensorflow"], default="torch")
    parser.add_argument("--dataset", choices=["synthetic", "cifar10_catsdogs"], default="synthetic")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download-data", action="store_true")
    args = parser.parse_args()

    trainer_cmd, worker_cmd = _print_commands(
        backend=args.backend,
        dataset=args.dataset,
        data_root=args.data_root,
        download_data=args.download_data,
    )

    if not args.launch:
        return

    repo_root = Path(__file__).resolve().parents[2]
    procs = [
        _launch_process(BACKEND_CMD, repo_root),
        _launch_process(trainer_cmd, repo_root),
        _launch_process(FORECAST_CMDS[args.backend], repo_root),
        _launch_process(worker_cmd, repo_root),
    ]

    print(f"\nLaunched 4 processes for {args.duration}s. Press Ctrl+C to stop early.")
    try:
        time.sleep(max(1, args.duration))
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs:
            proc.send_signal(signal.SIGTERM)
        for proc in procs:
            proc.wait(timeout=10)
        print("Processes stopped.")


if __name__ == "__main__":
    main()

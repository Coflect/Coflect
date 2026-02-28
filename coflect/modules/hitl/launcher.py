"""One-command launcher for the full Coflect HITL stack."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for launching HITL processes."""

    backend: str
    host: str
    port: int
    steps: int
    xai_every: int
    forecast_every: int
    snapshot_every: int
    batch_size: int
    trainer_device: str
    xai_device: str
    dataset: str
    data_root: str
    split: str
    download_data: bool
    num_workers: int
    xai_method: str
    xai_budget_ms_per_minute: float
    snapshot_dir: str
    log_dir: str
    startup_wait_s: float


@dataclass(frozen=True)
class ProcessSpec:
    """Command specification for one child process."""

    name: str
    module: str
    args: list[str]


@dataclass
class RunningProcess:
    """Live process handle with metadata and log destination."""

    spec: ProcessSpec
    proc: subprocess.Popen[bytes]
    log_path: Path
    log_handle: object


def _server_host_for_clients(host: str) -> str:
    """Normalize listener host to a client-reachable host for local workers."""
    normalized = host.strip()
    if normalized in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return normalized


def _build_process_specs(cfg: RunConfig) -> list[ProcessSpec]:
    """Build ordered process specs for backend + trainer + workers."""
    server = f"http://{_server_host_for_clients(cfg.host)}:{cfg.port}"
    snapshot_dir = cfg.snapshot_dir.strip() or ("snapshots_tf" if cfg.backend == "tensorflow" else "snapshots")

    specs: list[ProcessSpec] = [
        ProcessSpec(
            name="backend",
            module="coflect.modules.hitl.backend.app",
            args=["--host", cfg.host, "--port", str(cfg.port)],
        ),
        ProcessSpec(
            name="forecast",
            module="coflect.modules.hitl.forecast.worker",
            args=["--server", server, "--backend", cfg.backend],
        ),
    ]

    if cfg.backend == "torch":
        trainer_args = [
            "--server",
            server,
            "--steps",
            str(cfg.steps),
            "--xai_every",
            str(cfg.xai_every),
            "--forecast_every",
            str(cfg.forecast_every),
            "--snapshot_every",
            str(cfg.snapshot_every),
            "--batch_size",
            str(cfg.batch_size),
            "--num_workers",
            str(cfg.num_workers),
            "--dataset",
            cfg.dataset,
            "--data_root",
            cfg.data_root,
            "--split",
            cfg.split,
            "--snapshot_dir",
            snapshot_dir,
        ]
        if cfg.download_data:
            trainer_args.append("--download_data")
        if cfg.trainer_device.strip():
            trainer_args.extend(["--device", cfg.trainer_device.strip()])

        xai_args = [
            "--server",
            server,
            "--xai_method",
            cfg.xai_method,
            "--dataset",
            cfg.dataset,
            "--data_root",
            cfg.data_root,
            "--split",
            cfg.split,
            "--snapshot_dir",
            snapshot_dir,
            "--xai_budget_ms_per_minute",
            str(cfg.xai_budget_ms_per_minute),
        ]
        if cfg.download_data:
            xai_args.append("--download_data")
        if cfg.xai_device.strip():
            xai_args.extend(["--device", cfg.xai_device.strip()])

        specs.append(
            ProcessSpec(
                name="xai_worker",
                module="coflect.modules.hitl.xai_worker.worker_torch_livecam",
                args=xai_args,
            )
        )
        specs.append(
            ProcessSpec(
                name="trainer",
                module="coflect.modules.hitl.trainer.train_torch",
                args=trainer_args,
            )
        )
        return specs

    trainer_args = [
        "--server",
        server,
        "--steps",
        str(cfg.steps),
        "--xai_every",
        str(cfg.xai_every),
        "--forecast_every",
        str(cfg.forecast_every),
        "--snapshot_every",
        str(cfg.snapshot_every),
        "--batch_size",
        str(cfg.batch_size),
        "--snapshot_dir",
        snapshot_dir,
    ]
    if cfg.trainer_device.strip():
        trainer_args.extend(["--device", cfg.trainer_device.strip()])

    xai_args = [
        "--server",
        server,
        "--xai_method",
        cfg.xai_method,
        "--snapshot_dir",
        snapshot_dir,
        "--xai_budget_ms_per_minute",
        str(cfg.xai_budget_ms_per_minute),
    ]
    if cfg.xai_device.strip():
        xai_args.extend(["--device", cfg.xai_device.strip()])

    specs.append(
        ProcessSpec(
            name="xai_worker",
            module="coflect.modules.hitl.xai_worker.worker_tf_livecam",
            args=xai_args,
        )
    )
    specs.append(
        ProcessSpec(
            name="trainer",
            module="coflect.modules.hitl.trainer.train_tf",
            args=trainer_args,
        )
    )
    return specs


def _log_path(log_dir: Path, name: str) -> Path:
    """Return a per-process log path under the chosen log directory."""
    return log_dir / f"{name}.log"


def _launch_processes(specs: list[ProcessSpec], log_dir: Path, startup_wait_s: float) -> list[RunningProcess]:
    """Launch child processes and return running handles."""
    running: list[RunningProcess] = []
    for index, spec in enumerate(specs):
        log_path = _log_path(log_dir, spec.name)
        log_handle = log_path.open("wb")
        cmd = [sys.executable, "-m", spec.module, *spec.args]
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
        running.append(RunningProcess(spec=spec, proc=proc, log_path=log_path, log_handle=log_handle))
        print(f"[start] {spec.name} pid={proc.pid} log={log_path}")
        print(f"        {shlex.join(cmd)}")
        if index == 0 and startup_wait_s > 0:
            # Give backend time to bind before workers/trainers connect.
            time.sleep(startup_wait_s)
    return running


def _shutdown_processes(running: list[RunningProcess], timeout_s: float = 8.0) -> None:
    """Gracefully terminate all live processes, then force-kill if needed."""
    for rp in running:
        if rp.proc.poll() is None:
            rp.proc.terminate()

    deadline = time.time() + timeout_s
    for rp in running:
        remaining = max(0.0, deadline - time.time())
        if rp.proc.poll() is None:
            try:
                rp.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                rp.proc.kill()
                rp.proc.wait(timeout=2.0)

    for rp in running:
        rp.log_handle.close()


def _monitor(running: list[RunningProcess], poll_s: float = 0.5) -> int:
    """Monitor child processes and return exit code for launcher."""
    trainer = next(rp for rp in running if rp.spec.name == "trainer")
    critical_names = {"backend", "forecast", "xai_worker"}

    while True:
        trainer_code = trainer.proc.poll()
        if trainer_code is not None:
            if trainer_code == 0:
                print("[done] trainer finished; shutting down background workers.")
            else:
                print(f"[error] trainer exited with code {trainer_code}.")
            return int(trainer_code)

        for rp in running:
            if rp.spec.name in critical_names:
                code = rp.proc.poll()
                if code is not None:
                    print(f"[error] {rp.spec.name} exited with code {code}.")
                    return 1
        time.sleep(poll_s)


def _parse_args() -> RunConfig:
    """Parse CLI args into a structured run configuration."""
    ap = argparse.ArgumentParser(description="Launch backend + trainer + forecast + XAI in one command.")
    ap.add_argument("--backend", choices=["torch", "tensorflow"], default="torch")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--xai_every", type=int, default=250)
    ap.add_argument("--forecast_every", type=int, default=20)
    ap.add_argument("--snapshot_every", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--trainer_device", type=str, default="")
    ap.add_argument("--xai_device", type=str, default="")
    ap.add_argument("--dataset", choices=["synthetic", "cifar10_catsdogs"], default="synthetic")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--download_data", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument(
        "--xai_method",
        type=str,
        default="consensus",
        choices=["livecam", "gradcam", "smoothgrad", "consensus"],
    )
    ap.add_argument("--xai_budget_ms_per_minute", type=float, default=10_000.0)
    ap.add_argument("--snapshot_dir", type=str, default="")
    ap.add_argument("--log_dir", type=str, default="./.coflect_logs/hitl")
    ap.add_argument("--startup_wait_s", type=float, default=1.5)
    args = ap.parse_args()

    return RunConfig(
        backend=args.backend,
        host=args.host,
        port=args.port,
        steps=args.steps,
        xai_every=args.xai_every,
        forecast_every=args.forecast_every,
        snapshot_every=args.snapshot_every,
        batch_size=args.batch_size,
        trainer_device=args.trainer_device,
        xai_device=args.xai_device,
        dataset=args.dataset,
        data_root=args.data_root,
        split=args.split,
        download_data=bool(args.download_data),
        num_workers=args.num_workers,
        xai_method=args.xai_method,
        xai_budget_ms_per_minute=args.xai_budget_ms_per_minute,
        snapshot_dir=args.snapshot_dir,
        log_dir=args.log_dir,
        startup_wait_s=args.startup_wait_s,
    )


def main() -> None:
    """Entry-point for `coflect-hitl-run`."""
    cfg = _parse_args()
    log_dir = Path(cfg.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_process_specs(cfg)
    client_host = _server_host_for_clients(cfg.host)
    ui_url = f"http://{client_host}:{cfg.port}"
    print(f"[coflect-hitl-run] backend={cfg.backend} ui={ui_url}")
    print(f"[coflect-hitl-run] logs={log_dir}")

    running: list[RunningProcess] = []
    exit_code = 0
    try:
        running = _launch_processes(specs, log_dir=log_dir, startup_wait_s=cfg.startup_wait_s)
        print("[ready] stack started. press Ctrl+C to stop.")
        exit_code = _monitor(running)
    except KeyboardInterrupt:
        print("[stop] received Ctrl+C, shutting down.")
    finally:
        _shutdown_processes(running)

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

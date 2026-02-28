#!/usr/bin/env python3
"""Benchmark plain Torch training vs Coflect trainer overhead.

This benchmark intentionally disables heavy HITL paths in the Coflect run:
- no periodic XAI requests
- no forecast telemetry
- no mistake-triggered XAI
- no feedback polling
- no snapshots

It measures end-to-end training loop time for the same number of steps.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from coflect.modules.hitl.common.torch_dataset import DatasetConfig, build_torch_dataset


@dataclass(frozen=True)
class BenchConfig:
    repeats: int
    warmup_runs: int
    steps: int
    batch_size: int
    num_workers: int
    dataset: str
    data_root: str
    split: str
    download_data: bool
    device: str
    host: str
    port: int
    output: str


def _parse_args() -> BenchConfig:
    ap = argparse.ArgumentParser(description="Long-run benchmark: baseline Torch vs Coflect trainer.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup_runs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--dataset", choices=["synthetic", "cifar10_catsdogs"], default="synthetic")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--download_data", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8062)
    ap.add_argument(
        "--output",
        type=str,
        default="docs/benchmarks/hitl_overhead_longrun_latest.json",
        help="Path for JSON result artifact.",
    )
    args = ap.parse_args()
    return BenchConfig(
        repeats=max(1, args.repeats),
        warmup_runs=max(0, args.warmup_runs),
        steps=max(1, args.steps),
        batch_size=max(1, args.batch_size),
        num_workers=max(0, args.num_workers),
        dataset=str(args.dataset),
        data_root=str(args.data_root),
        split=str(args.split),
        download_data=bool(args.download_data),
        device=str(args.device),
        host=str(args.host),
        port=int(args.port),
        output=str(args.output),
    )


def _baseline_once(cfg: BenchConfig, run_seed: int) -> float:
    torch.manual_seed(run_seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(cfg.device)
    ds = build_torch_dataset(
        DatasetConfig(
            name=cfg.dataset,  # type: ignore[arg-type]
            root=cfg.data_root,
            split=cfg.split,  # type: ignore[arg-type]
            download_data=cfg.download_data,
        )
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    model = resnet18(num_classes=int(getattr(ds, "num_classes", 10))).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    it = iter(dl)
    t0 = time.perf_counter()
    for _ in range(cfg.steps):
        try:
            x, y, _idx = next(it)
        except StopIteration:
            it = iter(dl)
            x, y, _idx = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    t1 = time.perf_counter()
    return t1 - t0


def _trainer_cmd(cfg: BenchConfig) -> list[str]:
    server = f"http://{cfg.host}:{cfg.port}"
    cmd = [
        sys.executable,
        "-m",
        "coflect.modules.hitl.trainer.train_torch",
        "--server",
        server,
        "--device",
        cfg.device,
        "--steps",
        str(cfg.steps),
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
        "--xai_every",
        "1000000000",
        "--forecast_every",
        "0",
        "--mistake_every",
        "0",
        "--feedback_poll_every",
        "1000000000",
        "--snapshot_every",
        "1000000000",
    ]
    if cfg.download_data:
        cmd.append("--download_data")
    return cmd


def _coflect_once(cfg: BenchConfig) -> float:
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root if not env.get("PYTHONPATH") else f"{root}:{env['PYTHONPATH']}"
    cmd = _trainer_cmd(cfg)
    t0 = time.perf_counter()
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t1 = time.perf_counter()
    return t1 - t0


def _summary(elapsed: list[float], steps: int) -> dict[str, float]:
    sps = [steps / max(1e-9, e) for e in elapsed]
    return {
        "runs": float(len(elapsed)),
        "mean_elapsed_s": float(statistics.fmean(elapsed)),
        "stdev_elapsed_s": float(statistics.stdev(elapsed)) if len(elapsed) > 1 else 0.0,
        "median_elapsed_s": float(statistics.median(elapsed)),
        "mean_sps": float(statistics.fmean(sps)),
        "stdev_sps": float(statistics.stdev(sps)) if len(sps) > 1 else 0.0,
        "median_sps": float(statistics.median(sps)),
    }


def _start_backend(cfg: BenchConfig) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root if not env.get("PYTHONPATH") else f"{root}:{env['PYTHONPATH']}"
    cmd = [
        sys.executable,
        "-m",
        "coflect.modules.hitl.backend.app",
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    return proc


def _stop_backend(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)


def main() -> None:
    cfg = _parse_args()
    output_path = Path(cfg.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = _start_backend(cfg)
    if backend.poll() is not None:
        raise RuntimeError("Backend failed to start for Coflect benchmark run.")

    baseline_elapsed: list[float] = []
    coflect_elapsed: list[float] = []

    try:
        for i in range(cfg.warmup_runs):
            _ = _baseline_once(cfg, run_seed=10_000 + i)
            _ = _coflect_once(cfg)

        for i in range(cfg.repeats):
            b = _baseline_once(cfg, run_seed=20_000 + i)
            c = _coflect_once(cfg)
            baseline_elapsed.append(b)
            coflect_elapsed.append(c)
            print(
                f"run {i + 1}/{cfg.repeats}: "
                f"baseline={b:.3f}s ({cfg.steps / b:.3f} sps), "
                f"coflect={c:.3f}s ({cfg.steps / c:.3f} sps)"
            )
    finally:
        _stop_backend(backend)

    base = _summary(baseline_elapsed, cfg.steps)
    cof = _summary(coflect_elapsed, cfg.steps)
    slowdown_pct = (cof["mean_elapsed_s"] / max(1e-9, base["mean_elapsed_s"]) - 1.0) * 100.0
    sps_delta_pct = (cof["mean_sps"] / max(1e-9, base["mean_sps"]) - 1.0) * 100.0

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "config": {
            "repeats": cfg.repeats,
            "warmup_runs": cfg.warmup_runs,
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "dataset": cfg.dataset,
            "data_root": cfg.data_root,
            "split": cfg.split,
            "download_data": cfg.download_data,
            "device": cfg.device,
            "backend_host": cfg.host,
            "backend_port": cfg.port,
        },
        "baseline": {**base, "elapsed_s_per_run": baseline_elapsed},
        "coflect_minimal": {**cof, "elapsed_s_per_run": coflect_elapsed},
        "comparison": {
            "slowdown_pct_elapsed_mean": float(slowdown_pct),
            "delta_pct_sps_mean": float(sps_delta_pct),
        },
    }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()

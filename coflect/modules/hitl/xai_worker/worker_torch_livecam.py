"""Asynchronous Torch LiveCAM worker for attribution generation and event publishing."""

from __future__ import annotations

import argparse
import base64
import glob
import os
import re
import time
from collections import deque

import torch
import torch.nn.functional as F
from torchvision.models import resnet18

from coflect.backends.torch_backend import TorchAdapter
from coflect.modules.hitl.common.torch_dataset import DatasetConfig, build_torch_dataset
from coflect.modules.hitl.common.wire import dequeue_xai, post_event
from coflect.modules.hitl.xai_worker.livecam import make_overlay_png_with_meta

_SNAP_RE = re.compile(r"model_step_(\d+)\.pt$")


def _latest_snapshot_path(snapshot_dir: str) -> tuple[int, str] | None:
    """Return latest trainer snapshot path by step number."""
    latest_step = -1
    latest_path = ""
    for path in glob.glob(os.path.join(snapshot_dir, "model_step_*.pt")):
        m = _SNAP_RE.search(os.path.basename(path))
        if m is None:
            continue
        step = int(m.group(1))
        if step > latest_step:
            latest_step = step
            latest_path = path
    if latest_step < 0:
        return None
    return latest_step, latest_path


def _prune_budget(q: deque[tuple[float, float]], now: float, horizon_s: float = 60.0) -> float:
    """Drop expired timing entries and return rolling millisecond usage."""
    while q and (now - q[0][0]) > horizon_s:
        q.popleft()
    return sum(ms for _, ms in q)


def main() -> None:
    """Run LiveCAM worker loop with optional attribution budget limiting."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, default="http://localhost:8000")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--poll", type=float, default=0.2)
    ap.add_argument("--dataset", choices=["synthetic", "cifar10_catsdogs"], default="synthetic")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--download_data", action="store_true")
    ap.add_argument("--snapshot_dir", type=str, default="snapshots")
    ap.add_argument("--xai_budget_ms_per_minute", type=float, default=10_000.0)
    ap.add_argument(
        "--xai_method",
        type=str,
        default="consensus",
        choices=["livecam", "gradcam", "smoothgrad", "consensus"],
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    ds = build_torch_dataset(
        DatasetConfig(
            name=args.dataset,
            root=args.data_root,
            split=args.split,
            download_data=args.download_data,
        )
    )
    num_classes = int(getattr(ds, "num_classes", 10))

    model = resnet18(num_classes=num_classes).to(device)
    model.eval()
    adapter = TorchAdapter(model=model, optimizer=None, criterion=None)
    loaded_model_step = -1

    xai_times: deque[tuple[float, float]] = deque()

    post_event(args.server, "xai_worker", {
        "status": "online",
        "backend": "torch",
        "device": str(device),
        "snapshot_dir": args.snapshot_dir,
        "xai_budget_ms_per_minute": float(args.xai_budget_ms_per_minute),
        "xai_method": args.xai_method,
    })

    while True:
        item = dequeue_xai(args.server, timeout_s=2.0, backend="torch")
        if item is None:
            time.sleep(args.poll)
            continue

        step = int(item["step"])
        sample_idx = int(item["sample_idx"])
        target_class = int(item["target_class"])
        pred_class = int(item["pred_class"])
        request_kind = str(item.get("request_kind", "periodic"))
        risk_score = item.get("risk_score", None)
        horizon_epochs = item.get("horizon_epochs", None)

        latest = _latest_snapshot_path(args.snapshot_dir)
        if latest is not None:
            latest_step, latest_path = latest
            if latest_step > loaded_model_step:
                try:
                    state = torch.load(latest_path, map_location=device)
                    adapter.model.load_state_dict(state, strict=True)
                    adapter.model.eval()
                    loaded_model_step = latest_step
                    post_event(args.server, "xai_model_loaded", {
                        "worker_step": step,
                        "backend": "torch",
                        "model_step": loaded_model_step,
                        "snapshot_path": latest_path,
                    })
                except Exception as e:
                    post_event(args.server, "xai_model_load_error", {
                        "worker_step": step,
                        "backend": "torch",
                        "snapshot_path": latest_path,
                        "error": str(e),
                    })

        now = time.time()
        budget_used = _prune_budget(xai_times, now=now)
        if budget_used >= args.xai_budget_ms_per_minute:
            post_event(args.server, "xai_budget_skip", {
                "step": step,
                "backend": "torch",
                "sample_idx": sample_idx,
                "budget_ms_per_minute": float(args.xai_budget_ms_per_minute),
                "budget_used_ms": float(budget_used),
            })
            continue

        x, _, _ = ds[sample_idx]
        x = x.to(device)

        try:
            with torch.no_grad():
                logits = adapter.model(x.unsqueeze(0))
                probs = F.softmax(logits, dim=1)[0]
                topk = torch.topk(probs, k=min(3, probs.numel()))
                top_classes = [int(i) for i in topk.indices.tolist()]
                top_probs = [float(v) for v in topk.values.tolist()]

            t_xai0 = time.time()
            png, meta = make_overlay_png_with_meta(
                adapter.model,
                x,
                target_class=target_class,
                method=args.xai_method,
            )
            xai_ms = (time.time() - t_xai0) * 1000.0
            xai_times.append((time.time(), xai_ms))
            b64 = base64.b64encode(png).decode("utf-8")
            modality_focus = {"image": 1.0}
            post_event(args.server, "xai_image", {
                "step": step,
                "backend": "torch",
                "model_step": loaded_model_step,
                "sample_idx": sample_idx,
                "target_class": target_class,
                "pred_class": pred_class,
                "request_kind": request_kind,
                "risk_score": risk_score,
                "horizon_epochs": horizon_epochs,
                "modality_focus": modality_focus,
                "xai_method": args.xai_method,
                "top_classes": top_classes,
                "top_probs": top_probs,
                **meta,
                "xai_ms": float(xai_ms),
                "png_b64": b64
            })
            post_event(args.server, "xai_timing", {
                "step": step,
                "backend": "torch",
                "sample_idx": sample_idx,
                "request_kind": request_kind,
                "risk_score": risk_score,
                "horizon_epochs": horizon_epochs,
                "modality_focus": modality_focus,
                "xai_method": args.xai_method,
                **meta,
                "xai_ms": float(xai_ms),
                "model_step": loaded_model_step,
            })
        except Exception as e:
            post_event(args.server, "xai_error", {
                "step": step,
                "backend": "torch",
                "sample_idx": sample_idx,
                "error": str(e)
            })


if __name__ == "__main__":
    main()

"""Asynchronous TensorFlow LiveCAM worker for attribution generation and event publishing."""

from __future__ import annotations

import argparse
import base64
import glob
import os
import re
import time
from collections import deque

import numpy as np

from coflect.modules.hitl.common.synth_numpy import sample_chw_by_idx
from coflect.modules.hitl.common.tf_model import build_tf_cnn
from coflect.modules.hitl.common.wire import dequeue_xai, post_event
from coflect.modules.hitl.xai_worker.livecam_tf import (
    make_overlay_png_tf_with_meta,
)

_SNAP_RE = re.compile(r"model_step_(\d+)\.npz$")
_ARR_RE = re.compile(r"arr_(\d+)$")


def _latest_snapshot_path(snapshot_dir: str) -> tuple[int, str] | None:
    """Return latest trainer snapshot path by step number."""
    latest_step = -1
    latest_path = ""
    for path in glob.glob(os.path.join(snapshot_dir, "model_step_*.npz")):
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


def _load_snapshot_weights(path: str) -> list[np.ndarray]:
    """Load npz snapshot weights preserving original variable order."""
    with np.load(path) as data:
        keys = list(data.files)

        def _arr_key(name: str) -> int:
            """Sort `arr_<idx>` keys numerically; unknown keys go last.

            Example:
                >>> _arr_key("arr_3")
                3
            """
            m = _ARR_RE.match(name)
            return int(m.group(1)) if m is not None else 10**9

        keys.sort(key=_arr_key)
        return [np.array(data[k]) for k in keys]


def _configure_tf_device(device: str) -> None:
    """Apply simple device preference without hard-failing on unsupported layouts."""
    import tensorflow as tf

    if device.strip().lower() == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            return


def main() -> None:
    """Run TensorFlow LiveCAM worker loop with optional attribution budget limiting."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("TensorFlow is required; install with `pip install coflect[tensorflow]`") from exc

    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, default="http://localhost:8000")
    ap.add_argument("--device", type=str, default="gpu" if tf.config.list_physical_devices("GPU") else "cpu")
    ap.add_argument("--poll", type=float, default=0.2)
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--snapshot_dir", type=str, default="snapshots_tf")
    ap.add_argument("--xai_budget_ms_per_minute", type=float, default=10_000.0)
    ap.add_argument(
        "--xai_method",
        type=str,
        default="consensus",
        choices=["livecam", "gradcam", "smoothgrad", "consensus"],
    )
    args = ap.parse_args()

    _configure_tf_device(args.device)

    model = build_tf_cnn(num_classes=args.num_classes, image_size=args.image_size)
    loaded_model_step = -1
    xai_times: deque[tuple[float, float]] = deque()

    post_event(
        args.server,
        "xai_worker",
        {
            "status": "online",
            "backend": "tensorflow",
            "device": args.device,
            "snapshot_dir": args.snapshot_dir,
            "xai_budget_ms_per_minute": float(args.xai_budget_ms_per_minute),
            "xai_method": args.xai_method,
        },
    )

    while True:
        item = dequeue_xai(args.server, timeout_s=2.0, backend="tensorflow")
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
                    weights = _load_snapshot_weights(latest_path)
                    model.set_weights(weights)
                    loaded_model_step = latest_step
                    post_event(
                        args.server,
                        "xai_model_loaded",
                        {
                            "worker_step": step,
                            "backend": "tensorflow",
                            "model_step": loaded_model_step,
                            "snapshot_path": latest_path,
                        },
                    )
                except Exception as e:
                    post_event(
                        args.server,
                        "xai_model_load_error",
                        {
                            "worker_step": step,
                            "backend": "tensorflow",
                            "snapshot_path": latest_path,
                            "error": str(e),
                        },
                    )

        now = time.time()
        budget_used = _prune_budget(xai_times, now=now)
        if budget_used >= args.xai_budget_ms_per_minute:
            post_event(
                args.server,
                "xai_budget_skip",
                {
                    "step": step,
                    "backend": "tensorflow",
                    "sample_idx": sample_idx,
                    "budget_ms_per_minute": float(args.xai_budget_ms_per_minute),
                    "budget_used_ms": float(budget_used),
                },
            )
            continue

        x_chw, _ = sample_chw_by_idx(
            sample_idx,
            num_classes=args.num_classes,
            image_size=args.image_size,
        )
        x_nhwc = np.transpose(x_chw, (1, 2, 0))[np.newaxis, ...].astype(np.float32)

        try:
            logits = np.asarray(model(x_nhwc, training=False))[0]
            logits = logits.astype(np.float64, copy=False)
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / (np.sum(probs) + 1e-12)
            top_order = np.argsort(probs)[::-1][: min(3, probs.shape[0])]
            top_classes = [int(i) for i in top_order.tolist()]
            top_probs = [float(probs[i]) for i in top_order.tolist()]

            t_xai0 = time.time()
            png, meta = make_overlay_png_tf_with_meta(
                model,
                x_chw,
                target_class=target_class,
                method=args.xai_method,
            )
            xai_ms = (time.time() - t_xai0) * 1000.0
            xai_times.append((time.time(), xai_ms))
            b64 = base64.b64encode(png).decode("utf-8")
            modality_focus = {"image": 1.0}
            post_event(
                args.server,
                "xai_image",
                {
                    "step": step,
                    "backend": "tensorflow",
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
                    "png_b64": b64,
                },
            )
            post_event(
                args.server,
                "xai_timing",
                {
                    "step": step,
                    "backend": "tensorflow",
                    "sample_idx": sample_idx,
                    "request_kind": request_kind,
                    "risk_score": risk_score,
                    "horizon_epochs": horizon_epochs,
                    "modality_focus": modality_focus,
                    "xai_method": args.xai_method,
                    **meta,
                    "xai_ms": float(xai_ms),
                    "model_step": loaded_model_step,
                },
            )
        except Exception as e:
            post_event(
                args.server,
                "xai_error",
                {
                    "step": step,
                    "backend": "tensorflow",
                    "sample_idx": sample_idx,
                    "error": str(e),
                },
            )


if __name__ == "__main__":
    main()

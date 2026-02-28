"""PyTorch trainer loop for HITL MVP with asynchronous XAI orchestration."""

from __future__ import annotations

import argparse
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from coflect.backends.torch_backend import TorchAdapter
from coflect.modules.hitl.common.instruction_parser import parse_instruction
from coflect.modules.hitl.common.messages import XaiRequestPayload
from coflect.modules.hitl.common.torch_dataset import DatasetConfig, build_torch_dataset
from coflect.modules.hitl.common.wire import (
    enqueue_forecast_telemetry,
    enqueue_xai,
    get_feedback,
    post_event,
)

_ROI_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _clamp01(x: float) -> float:
    """Clamp numeric value into [0.0, 1.0].

    Example:
        >>> _clamp01(1.7)
        1.0
    """
    return max(0.0, min(1.0, x))


@dataclass(frozen=True)
class AuxConfig:
    """Configuration for cheap auxiliary ROI-alignment loss cadence."""

    every: int
    subset: int


class AsyncSnapshotWriter:
    """Background snapshot writer using atomic rename semantics."""

    def __init__(self, out_dir: str):
        """Start asynchronous snapshot writer thread.

        Example:
            >>> writer = AsyncSnapshotWriter(\"snapshots\")
        """
        self.out_dir = out_dir
        # Keep queue size tiny so stale snapshots are dropped before disk write.
        self._queue: queue.Queue[tuple[int, dict[str, torch.Tensor]]] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._closed = False
        os.makedirs(self.out_dir, exist_ok=True)
        self._thread.start()

    def enqueue(self, step: int, state: dict[str, torch.Tensor]) -> None:
        """Queue latest model state for async atomic disk write.

        Example:
            >>> writer.enqueue(step=100, state=model.state_dict())
        """
        if self._closed:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait((step, state))
        except queue.Full:
            return

    def close(self) -> None:
        """Stop background writer thread and flush termination sentinel.

        Example:
            >>> writer.close()
        """
        self._closed = True
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait((-1, {}))
        except queue.Full:
            pass
        self._thread.join(timeout=10.0)

    def _run(self) -> None:
        """Worker loop that writes snapshots using temp-file replace."""
        while True:
            step, state = self._queue.get()
            if step < 0:
                return
            tmp_path = os.path.join(self.out_dir, f".model_step_{step:06d}.pt.tmp")
            final_path = os.path.join(self.out_dir, f"model_step_{step:06d}.pt")
            torch.save(state, tmp_path)
            os.replace(tmp_path, final_path)


def _parse_roi(feedback: dict[str, Any], h: int, w: int) -> tuple[int, int, int, int] | None:
    """Parse ROI from feedback in either normalized [0,1] or pixel coordinates."""
    roi = feedback.get("roi")
    if isinstance(roi, dict):
        vals = [roi.get("x0"), roi.get("y0"), roi.get("x1"), roi.get("y1")]
    elif isinstance(roi, (list, tuple)) and len(roi) == 4:
        vals = list(roi)
    else:
        return None

    if any(v is None for v in vals):
        return None

    parsed: list[float] = []
    for v in vals:
        if isinstance(v, (int, float)):
            parsed.append(float(v))
            continue
        if isinstance(v, str) and _ROI_NUMERIC_RE.match(v.strip()):
            parsed.append(float(v.strip()))
            continue
        return None

    x0f, y0f, x1f, y1f = parsed
    if max(abs(x0f), abs(y0f), abs(x1f), abs(y1f)) <= 1.5:
        x0f, x1f = x0f * w, x1f * w
        y0f, y1f = y0f * h, y1f * h

    x0 = int(max(0, min(w - 1, min(x0f, x1f))))
    x1 = int(max(1, min(w, max(x0f, x1f))))
    y0 = int(max(0, min(h - 1, min(y0f, y1f))))
    y1 = int(max(1, min(h, max(y0f, y1f))))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _roi_mask(shape_hw: tuple[int, int], roi: tuple[int, int, int, int] | None, device: torch.device) -> torch.Tensor | None:
    """Create a binary ROI mask tensor from parsed box coordinates."""
    if roi is None:
        return None
    h, w = shape_hw
    x0, y0, x1, y1 = roi
    mask = torch.zeros((1, 1, h, w), device=device)
    mask[:, :, y0:y1, x0:x1] = 1.0
    return mask


def _roi_norm_to_pixels(roi_norm: tuple[float, float, float, float], h: int, w: int) -> tuple[int, int, int, int] | None:
    """Convert normalized ROI `(x0,y0,x1,y1)` to bounded pixel coordinates.

    Example:
        >>> _roi_norm_to_pixels((0.25, 0.25, 0.75, 0.75), h=64, w=64)
    """
    x0n, y0n, x1n, y1n = roi_norm
    x0 = int(max(0, min(w - 1, min(x0n, x1n) * w)))
    x1 = int(max(1, min(w, max(x0n, x1n) * w)))
    y0 = int(max(0, min(h - 1, min(y0n, y1n) * h)))
    y1 = int(max(1, min(h, max(y0n, y1n) * h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _parse_roi_mask(feedback: dict[str, Any], h: int, w: int) -> torch.Tensor | None:
    """Parse a dense ROI mask payload in flattened or HxW list form."""
    raw = feedback.get("roi_mask")
    if raw is None:
        return None

    if isinstance(raw, list) and len(raw) == h and all(isinstance(r, list) and len(r) == w for r in raw):
        try:
            mask = torch.tensor(raw, dtype=torch.float32)
        except (TypeError, ValueError):
            return None
        return mask.clamp_(0.0, 1.0).unsqueeze(0).unsqueeze(0).contiguous()

    if isinstance(raw, list) and len(raw) == h * w:
        try:
            mask = torch.tensor(raw, dtype=torch.float32).view(1, 1, h, w)
        except (TypeError, ValueError):
            return None
        return mask.clamp_(0.0, 1.0).contiguous()

    return None


def main() -> None:
    """Run training loop and coordinate async feedback/XAI/snapshot interactions."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, default="http://localhost:8000")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dataset", choices=["synthetic", "cifar10_catsdogs"], default="synthetic")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--download_data", action="store_true")
    ap.add_argument("--xai_every", type=int, default=250)          # request XAI every N steps
    ap.add_argument("--feedback_poll_every", type=int, default=50)  # poll feedback every N steps
    ap.add_argument("--snapshot_every", type=int, default=500)
    ap.add_argument("--snapshot_dir", type=str, default="snapshots")
    ap.add_argument("--aux_every", type=int, default=50)
    ap.add_argument("--aux_subset", type=int, default=8)
    ap.add_argument("--mistake_every", type=int, default=40)
    ap.add_argument("--forecast_every", type=int, default=20)
    ap.add_argument("--telemetry_samples", type=int, default=2)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    ds = build_torch_dataset(
        DatasetConfig(
            name=args.dataset,
            root=args.data_root,
            split=args.split,
            download_data=args.download_data,
        )
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = max(1, len(dl))
    num_classes = int(getattr(ds, "num_classes", 10))

    model = resnet18(num_classes=num_classes).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    adapter = TorchAdapter(model=model, optimizer=opt, criterion=criterion)
    aux_cfg = AuxConfig(every=max(1, args.aux_every), subset=max(1, args.aux_subset))

    focus_lambda = 0.0
    current_roi: tuple[int, int, int, int] | None = None
    current_roi_mask_cpu: torch.Tensor | None = None
    sample_rules: dict[int, tuple[tuple[int, int, int, int] | None, torch.Tensor | None]] = {}
    last_feedback: dict[str, Any] = {}
    paused = False

    it = iter(dl)
    t0 = time.time()
    last_metric_t = t0
    last_metric_step = 0
    snapshot_writer = AsyncSnapshotWriter(args.snapshot_dir)
    last_aux_loss: float | None = None
    pause_state_reported = False

    def _apply_feedback(fb: dict[str, Any], step: int, h: int, w: int) -> None:
        """Apply latest UI feedback to pause/ROI/focus policy in-place.

        Example:
            >>> _apply_feedback({\"instruction\": \"focus=0.3\"}, step=50, h=64, w=64)
        """
        nonlocal current_roi, current_roi_mask_cpu, focus_lambda, last_feedback, paused
        if not fb or fb == last_feedback:
            return
        last_feedback = fb
        raw_paused = fb.get("paused", None)
        if raw_paused is not None:
            paused = bool(raw_paused)

        instruction = str(fb.get("instruction", ""))
        parsed = parse_instruction(instruction)

        raw_strength = fb.get("strength", None)
        try:
            base_strength = float(raw_strength) if raw_strength is not None else focus_lambda
        except (TypeError, ValueError):
            base_strength = focus_lambda
        focus_lambda = _clamp01(base_strength + parsed.strength_delta)
        if parsed.strength is not None:
            focus_lambda = parsed.strength

        roi_updated = bool(fb.get("roi", None) is not None or fb.get("roi_mask", None) is not None or parsed.roi_norm is not None)
        if roi_updated:
            current_roi_mask_cpu = _parse_roi_mask(fb, h=h, w=w)
            current_roi = _parse_roi(fb, h=h, w=w)
            if current_roi is None and parsed.roi_norm is not None:
                current_roi = _roi_norm_to_pixels(parsed.roi_norm, h=h, w=w)
        sample_idx_for_rule: int | None = None
        raw_sample_idx = fb.get("sample_idx", None)
        if isinstance(raw_sample_idx, int):
            sample_idx_for_rule = raw_sample_idx
        elif isinstance(raw_sample_idx, str) and raw_sample_idx.strip().isdigit():
            sample_idx_for_rule = int(raw_sample_idx.strip())
        if (
            sample_idx_for_rule is not None
            and sample_idx_for_rule >= 0
            and roi_updated
            and (current_roi is not None or current_roi_mask_cpu is not None)
        ):
            if sample_idx_for_rule in sample_rules:
                sample_rules.pop(sample_idx_for_rule)
            sample_rules[sample_idx_for_rule] = (
                current_roi,
                current_roi_mask_cpu.clone() if current_roi_mask_cpu is not None else None,
            )
            while len(sample_rules) > 10:
                oldest = next(iter(sample_rules))
                sample_rules.pop(oldest)
        post_event(
            args.server,
            "trainer_feedback_applied",
            {
                "step": step,
                "backend": "torch",
                "paused": paused,
                "new_focus_lambda": focus_lambda,
                "roi": current_roi,
                "sample_rule_count": len(sample_rules),
                "instruction_policy": parsed.to_dict(),
                "roi_mask_applied": current_roi_mask_cpu is not None,
                "feedback": fb,
            },
        )

    try:
        for step in range(1, args.steps + 1):
            if paused:
                if not pause_state_reported:
                    post_event(args.server, "trainer_paused", {"step": step, "backend": "torch"})
                    pause_state_reported = True
                while paused:
                    time.sleep(0.25)
                    _apply_feedback(get_feedback(args.server), step=step, h=ds.image_size, w=ds.image_size)
                post_event(args.server, "trainer_resumed", {"step": step, "backend": "torch"})
                pause_state_reported = False

            try:
                x, y, idx = next(it)
            except StopIteration:
                it = iter(dl)
                x, y, idx = next(it)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = adapter.forward(x)
            primary_loss = adapter.loss(logits, y)
            total_loss = primary_loss

            if (
                focus_lambda > 0.0
                and step % aux_cfg.every == 0
                and (current_roi is not None or current_roi_mask_cpu is not None or bool(sample_rules))
            ):
                subset = min(aux_cfg.subset, x.shape[0])
                h, w = x.shape[2], x.shape[3]
                global_mask = None
                if current_roi_mask_cpu is not None:
                    global_mask = current_roi_mask_cpu.to(device=device, non_blocking=True)
                else:
                    global_mask = _roi_mask((h, w), current_roi, device)

                mask = torch.zeros((subset, 1, h, w), device=device)
                has_any_mask = False
                subset_indices = idx[:subset].tolist()
                for i, sid_raw in enumerate(subset_indices):
                    sid = int(sid_raw)
                    rule = sample_rules.get(sid, None)
                    rule_roi = rule[0] if rule is not None else None
                    rule_mask_cpu = rule[1] if rule is not None else None
                    if rule_mask_cpu is not None:
                        mask[i : i + 1] = rule_mask_cpu.to(device=device, non_blocking=True)
                        has_any_mask = True
                    elif rule_roi is not None:
                        roi_mask = _roi_mask((h, w), rule_roi, device)
                        if roi_mask is not None:
                            mask[i : i + 1] = roi_mask
                            has_any_mask = True
                    elif global_mask is not None:
                        mask[i : i + 1] = global_mask
                        has_any_mask = True
                if has_any_mask:
                    masked_x = x[:subset] * mask
                    aux_logits = adapter.forward(masked_x)
                    aux_loss_t = adapter.loss(aux_logits, y[:subset])
                    total_loss = total_loss + (focus_lambda * aux_loss_t)
                    last_aux_loss = float(aux_loss_t.detach().item())

            adapter.step(total_loss)

            if step % 10 == 0:
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    acc = (pred == y).float().mean().item()
                now = time.time()
                elapsed_total = max(1e-6, now - t0)
                elapsed_window = max(1e-6, now - last_metric_t)
                step_window = step - last_metric_step
                metrics_payload = {
                    "step": step,
                    "backend": "torch",
                    "loss": float(primary_loss.detach().item()),
                    "acc": float(acc),
                    "focus_lambda": float(focus_lambda),
                    "aux_loss": last_aux_loss,
                    "sps": float(step / elapsed_total),
                    "sps_window": float(step_window / elapsed_window),
                }
                post_event(args.server, "metrics", metrics_payload)
                last_metric_t = now
                last_metric_step = step

            if args.forecast_every > 0 and step % args.forecast_every == 0:
                with torch.no_grad():
                    m = max(1, min(args.telemetry_samples, x.shape[0]))
                    probs_m = torch.softmax(logits[:m], dim=1)
                    top2 = torch.topk(probs_m, k=min(2, probs_m.shape[1]), dim=1)
                    pred_m = top2.indices[:, 0]
                    if top2.values.shape[1] == 2:
                        margin_m = top2.values[:, 0] - top2.values[:, 1]
                    else:
                        margin_m = top2.values[:, 0]
                    p_true_m = probs_m.gather(1, y[:m].unsqueeze(1)).squeeze(1)
                    correct_m = pred_m.eq(y[:m])

                samples = []
                for i in range(m):
                    samples.append(
                        {
                            "sample_idx": int(idx[i]),
                            "target_class": int(y[i].item()),
                            "pred_class": int(pred_m[i].item()),
                            "p_true": float(p_true_m[i].item()),
                            "margin": float(margin_m[i].item()),
                            "correct": bool(correct_m[i].item()),
                        }
                    )
                enqueue_forecast_telemetry(
                    args.server,
                    {
                        "step": step,
                        "epoch": float(step / steps_per_epoch),
                        "samples": samples,
                    },
                    backend="torch",
                )

            if step % args.snapshot_every == 0:
                with torch.no_grad():
                    snap_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                snapshot_writer.enqueue(step, snap_state)
                post_event(
                    args.server,
                    "snapshot_queued",
                    {"step": step, "backend": "torch", "path_hint": args.snapshot_dir},
                )

            # async XAI request (send only indices + classes)
            if step % args.xai_every == 0:
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                sample_idx = int(idx[0])
                target_class = int(y[0].item())
                pred_class = int(pred[0].item())
                req = XaiRequestPayload(
                    step=step,
                    sample_idx=sample_idx,
                    target_class=target_class,
                    pred_class=pred_class,
                    request_kind="periodic",
                    backend="torch",
                )
                enqueue_xai(args.server, req.to_dict(), backend="torch")
                post_event(args.server, "xai_requested", req.to_dict())

            if args.mistake_every > 0 and step % args.mistake_every == 0:
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    wrong_idx = (pred != y).nonzero(as_tuple=False)
                if wrong_idx.numel() > 0:
                    i = int(wrong_idx[0].item())
                    req = XaiRequestPayload(
                        step=step,
                        sample_idx=int(idx[i]),
                        target_class=int(y[i].item()),
                        pred_class=int(pred[i].item()),
                        request_kind="mistake",
                        backend="torch",
                    )
                    enqueue_xai(args.server, req.to_dict(), backend="torch")
                    post_event(args.server, "xai_requested", req.to_dict())

            # lightweight feedback poll
            if step % args.feedback_poll_every == 0:
                _apply_feedback(get_feedback(args.server), step=step, h=x.shape[2], w=x.shape[3])
    finally:
        snapshot_writer.close()

    post_event(args.server, "done", {"steps": args.steps, "backend": "torch"})


if __name__ == "__main__":
    main()

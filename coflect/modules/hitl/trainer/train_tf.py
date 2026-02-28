"""TensorFlow/Keras trainer loop for HITL MVP with asynchronous XAI orchestration."""

from __future__ import annotations

import argparse
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from coflect.backends.tensorflow_backend import TensorFlowAdapter
from coflect.modules.hitl.common.instruction_parser import parse_instruction
from coflect.modules.hitl.common.messages import XaiRequestPayload
from coflect.modules.hitl.common.synth_numpy import chw_to_hwc, sample_chw_by_idx
from coflect.modules.hitl.common.tf_model import build_tf_cnn
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
        """Start asynchronous snapshot writer thread for `.npz` weights.

        Example:
            >>> writer = AsyncSnapshotWriter(\"snapshots_tf\")
        """
        self.out_dir = out_dir
        self._queue: queue.Queue[tuple[int, list[np.ndarray]]] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._closed = False
        os.makedirs(self.out_dir, exist_ok=True)
        self._thread.start()

    def enqueue(self, step: int, weights: list[np.ndarray]) -> None:
        """Queue latest model weights for async atomic write.

        Example:
            >>> writer.enqueue(step=100, weights=model.get_weights())
        """
        if self._closed:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait((step, weights))
        except queue.Full:
            return

    def close(self) -> None:
        """Stop background writer thread and enqueue termination sentinel.

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
            self._queue.put_nowait((-1, []))
        except queue.Full:
            pass
        self._thread.join(timeout=10.0)

    def _run(self) -> None:
        """Worker loop that writes `.npz` snapshots using temp-file replace."""
        while True:
            step, weights = self._queue.get()
            if step < 0:
                return
            tmp_path = os.path.join(self.out_dir, f".model_step_{step:06d}.npz.tmp")
            final_path = os.path.join(self.out_dir, f"model_step_{step:06d}.npz")
            with open(tmp_path, "wb") as fh:
                np.savez(fh, *weights)
            os.replace(tmp_path, final_path)


def _build_batch(
    sample_indices: np.ndarray,
    num_classes: int,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a deterministic NHWC batch by synthetic sample indices."""
    bsz = int(sample_indices.shape[0])
    x = np.empty((bsz, image_size, image_size, 3), dtype=np.float32)
    y = np.empty((bsz,), dtype=np.int32)
    for i, sidx in enumerate(sample_indices):
        img_chw, label = sample_chw_by_idx(
            int(sidx),
            num_classes=num_classes,
            image_size=image_size,
        )
        x[i] = chw_to_hwc(img_chw)
        y[i] = label
    return x, y


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


def _roi_mask(shape_hw: tuple[int, int], roi: tuple[int, int, int, int] | None) -> np.ndarray | None:
    """Create a binary ROI mask in NHWC broadcast shape from parsed box coordinates."""
    if roi is None:
        return None
    h, w = shape_hw
    x0, y0, x1, y1 = roi
    mask = np.zeros((1, h, w, 1), dtype=np.float32)
    mask[:, y0:y1, x0:x1, :] = 1.0
    return mask


def _roi_norm_to_pixels(
    roi_norm: tuple[float, float, float, float],
    h: int,
    w: int,
) -> tuple[int, int, int, int] | None:
    """Convert normalized ROI `(x0,y0,x1,y1)` to bounded pixel coordinates.

    Example:
        >>> _roi_norm_to_pixels((0.2, 0.2, 0.8, 0.8), h=64, w=64)
    """
    x0n, y0n, x1n, y1n = roi_norm
    x0 = int(max(0, min(w - 1, min(x0n, x1n) * w)))
    x1 = int(max(1, min(w, max(x0n, x1n) * w)))
    y0 = int(max(0, min(h - 1, min(y0n, y1n) * h)))
    y1 = int(max(1, min(h, max(y0n, y1n) * h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _parse_roi_mask(feedback: dict[str, Any], h: int, w: int) -> np.ndarray | None:
    """Parse a dense ROI mask payload in flattened or HxW list form."""
    raw = feedback.get("roi_mask")
    if raw is None:
        return None

    if isinstance(raw, list) and len(raw) == h and all(isinstance(r, list) and len(r) == w for r in raw):
        try:
            mask = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        return np.clip(mask, 0.0, 1.0)[np.newaxis, :, :, np.newaxis]

    if isinstance(raw, list) and len(raw) == h * w:
        try:
            mask = np.asarray(raw, dtype=np.float32).reshape(h, w)
        except (TypeError, ValueError):
            return None
        return np.clip(mask, 0.0, 1.0)[np.newaxis, :, :, np.newaxis]

    return None


def _configure_tf_device(device: str) -> None:
    """Apply simple device preference without hard-failing on unsupported layouts."""
    import tensorflow as tf

    if device.strip().lower() == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            return


def main() -> None:
    """Run TensorFlow training loop and coordinate async feedback/XAI/snapshot interactions."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("TensorFlow is required; install with `pip install coflect[tensorflow]`") from exc

    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, default="http://localhost:8000")
    ap.add_argument("--device", type=str, default="gpu" if tf.config.list_physical_devices("GPU") else "cpu")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--dataset_size", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--xai_every", type=int, default=250)
    ap.add_argument("--feedback_poll_every", type=int, default=50)
    ap.add_argument("--snapshot_every", type=int, default=500)
    ap.add_argument("--snapshot_dir", type=str, default="snapshots_tf")
    ap.add_argument("--aux_every", type=int, default=50)
    ap.add_argument("--aux_subset", type=int, default=8)
    ap.add_argument("--mistake_every", type=int, default=40)
    ap.add_argument("--forecast_every", type=int, default=20)
    ap.add_argument("--telemetry_samples", type=int, default=2)
    args = ap.parse_args()

    _configure_tf_device(args.device)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    model = build_tf_cnn(num_classes=args.num_classes, image_size=args.image_size)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    adapter = TensorFlowAdapter(model=model, optimizer=opt, loss_fn=loss_fn)
    aux_cfg = AuxConfig(every=max(1, args.aux_every), subset=max(1, args.aux_subset))

    rng = np.random.default_rng(seed=args.seed)
    steps_per_epoch = max(1.0, float(args.dataset_size) / float(args.batch_size))
    focus_lambda = 0.0
    current_roi: tuple[int, int, int, int] | None = None
    current_roi_mask_cpu: np.ndarray | None = None
    sample_rules: dict[int, tuple[tuple[int, int, int, int] | None, np.ndarray | None]] = {}
    last_feedback: dict[str, Any] = {}
    paused = False

    t0 = time.time()
    last_metric_t = t0
    last_metric_step = 0
    last_aux_loss: float | None = None
    snapshot_writer = AsyncSnapshotWriter(args.snapshot_dir)
    pause_state_reported = False

    def _apply_feedback(fb: dict[str, Any], step: int) -> None:
        """Apply latest UI feedback to pause/ROI/focus policy in-place.

        Example:
            >>> _apply_feedback({\"instruction\": \"focus=0.3\"}, step=50)
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
            current_roi_mask_cpu = _parse_roi_mask(
                fb,
                h=args.image_size,
                w=args.image_size,
            )
            current_roi = _parse_roi(fb, h=args.image_size, w=args.image_size)
            if current_roi is None and parsed.roi_norm is not None:
                current_roi = _roi_norm_to_pixels(
                    parsed.roi_norm,
                    h=args.image_size,
                    w=args.image_size,
                )
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
                np.array(current_roi_mask_cpu, copy=True) if current_roi_mask_cpu is not None else None,
            )
            while len(sample_rules) > 10:
                oldest = next(iter(sample_rules))
                sample_rules.pop(oldest)
        post_event(
            args.server,
            "trainer_feedback_applied",
            {
                "step": step,
                "backend": "tensorflow",
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
                    post_event(args.server, "trainer_paused", {"step": step, "backend": "tensorflow"})
                    pause_state_reported = True
                while paused:
                    time.sleep(0.25)
                    _apply_feedback(get_feedback(args.server), step=step)
                post_event(args.server, "trainer_resumed", {"step": step, "backend": "tensorflow"})
                pause_state_reported = False

            idx_batch = rng.integers(0, args.dataset_size, size=args.batch_size, dtype=np.int64)
            x_np, y_np = _build_batch(
                idx_batch,
                num_classes=args.num_classes,
                image_size=args.image_size,
            )
            x = tf.convert_to_tensor(x_np, dtype=tf.float32)
            y = tf.convert_to_tensor(y_np, dtype=tf.int32)

            with tf.GradientTape() as tape:
                logits = adapter.forward(x)
                primary_loss = adapter.loss(logits, y)
                total_loss = primary_loss

                if (
                    focus_lambda > 0.0
                    and step % aux_cfg.every == 0
                    and (current_roi is not None or current_roi_mask_cpu is not None or bool(sample_rules))
                ):
                    subset = min(aux_cfg.subset, args.batch_size)
                    global_mask = (
                        current_roi_mask_cpu
                        if current_roi_mask_cpu is not None
                        else _roi_mask((args.image_size, args.image_size), current_roi)
                    )
                    mask_batch = np.zeros((subset, args.image_size, args.image_size, 1), dtype=np.float32)
                    has_any_mask = False
                    for i in range(subset):
                        sid = int(idx_batch[i])
                        rule = sample_rules.get(sid, None)
                        rule_roi = rule[0] if rule is not None else None
                        rule_mask = rule[1] if rule is not None else None
                        if rule_mask is not None:
                            mask_batch[i : i + 1] = rule_mask
                            has_any_mask = True
                        elif rule_roi is not None:
                            roi_mask = _roi_mask((args.image_size, args.image_size), rule_roi)
                            if roi_mask is not None:
                                mask_batch[i : i + 1] = roi_mask
                                has_any_mask = True
                        elif global_mask is not None:
                            mask_batch[i : i + 1] = global_mask
                            has_any_mask = True
                    if has_any_mask:
                        mask_t = tf.convert_to_tensor(mask_batch, dtype=tf.float32)
                        masked_x = x[:subset] * mask_t
                        aux_logits = adapter.forward(masked_x)
                        aux_loss_t = adapter.loss(aux_logits, y[:subset])
                        total_loss = total_loss + (focus_lambda * aux_loss_t)
                        last_aux_loss = float(aux_loss_t.numpy())

            grads = tape.gradient(total_loss, adapter.model.trainable_variables)
            grads_and_vars = [
                (g, v)
                for g, v in zip(grads, adapter.model.trainable_variables, strict=False)
                if g is not None
            ]
            adapter.optimizer.apply_gradients(grads_and_vars)

            if step % 10 == 0:
                pred = tf.argmax(logits, axis=1, output_type=tf.int32)
                acc_t = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))
                now = time.time()
                elapsed_total = max(1e-6, now - t0)
                elapsed_window = max(1e-6, now - last_metric_t)
                step_window = step - last_metric_step
                metrics_payload = {
                    "step": step,
                    "backend": "tensorflow",
                    "loss": float(primary_loss.numpy()),
                    "acc": float(acc_t.numpy()),
                    "focus_lambda": float(focus_lambda),
                    "aux_loss": last_aux_loss,
                    "sps": float(step / elapsed_total),
                    "sps_window": float(step_window / elapsed_window),
                }
                post_event(args.server, "metrics", metrics_payload)
                last_metric_t = now
                last_metric_step = step

            if args.forecast_every > 0 and step % args.forecast_every == 0:
                logits_np = np.asarray(logits, dtype=np.float32)
                m = max(1, min(args.telemetry_samples, logits_np.shape[0]))
                logits_m = logits_np[:m]
                logits_m = logits_m - np.max(logits_m, axis=1, keepdims=True)
                probs_m = np.exp(logits_m)
                probs_m = probs_m / (np.sum(probs_m, axis=1, keepdims=True) + 1e-12)
                pred_m = np.argmax(probs_m, axis=1)
                top2 = np.sort(probs_m, axis=1)[:, -2:] if probs_m.shape[1] > 1 else probs_m
                if probs_m.shape[1] > 1:
                    margin_m = top2[:, 1] - top2[:, 0]
                else:
                    margin_m = top2[:, 0]
                y_m = y_np[:m]
                p_true_m = probs_m[np.arange(m), y_m]
                correct_m = pred_m == y_m
                samples = []
                for i in range(m):
                    samples.append(
                        {
                            "sample_idx": int(idx_batch[i]),
                            "target_class": int(y_m[i]),
                            "pred_class": int(pred_m[i]),
                            "p_true": float(p_true_m[i]),
                            "margin": float(margin_m[i]),
                            "correct": bool(correct_m[i]),
                        }
                    )
                enqueue_forecast_telemetry(
                    args.server,
                    {
                        "step": step,
                        "epoch": float(step / steps_per_epoch),
                        "samples": samples,
                    },
                    backend="tensorflow",
                )

            if step % args.snapshot_every == 0:
                snap_weights = [w.copy() for w in adapter.model.get_weights()]
                snapshot_writer.enqueue(step, snap_weights)
                post_event(
                    args.server,
                    "snapshot_queued",
                    {
                        "step": step,
                        "backend": "tensorflow",
                        "path_hint": args.snapshot_dir,
                    },
                )

            if step % args.xai_every == 0:
                pred_class = int(np.argmax(logits[0].numpy()))
                target_class = int(y_np[0])
                sample_idx = int(idx_batch[0])
                req = XaiRequestPayload(
                    step=step,
                    sample_idx=sample_idx,
                    target_class=target_class,
                    pred_class=pred_class,
                    request_kind="periodic",
                    backend="tensorflow",
                )
                enqueue_xai(args.server, req.to_dict(), backend="tensorflow")
                post_event(args.server, "xai_requested", req.to_dict())

            if args.mistake_every > 0 and step % args.mistake_every == 0:
                pred_np = np.asarray(tf.argmax(logits, axis=1), dtype=np.int64)
                wrong = np.where(pred_np != y_np)[0]
                if wrong.size > 0:
                    i = int(wrong[0])
                    req = XaiRequestPayload(
                        step=step,
                        sample_idx=int(idx_batch[i]),
                        target_class=int(y_np[i]),
                        pred_class=int(pred_np[i]),
                        request_kind="mistake",
                        backend="tensorflow",
                    )
                    enqueue_xai(args.server, req.to_dict(), backend="tensorflow")
                    post_event(args.server, "xai_requested", req.to_dict())

            if step % args.feedback_poll_every == 0:
                _apply_feedback(get_feedback(args.server), step=step)
    finally:
        snapshot_writer.close()

    post_event(args.server, "done", {"steps": args.steps, "backend": "tensorflow"})


if __name__ == "__main__":
    main()

"""CPU-only forecast worker that proposes top-k likely future failures."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import mean

from coflect.modules.hitl.common.messages import XaiRequestPayload
from coflect.modules.hitl.common.wire import (
    dequeue_forecast_telemetry,
    enqueue_xai,
    get_feedback,
    post_event,
    post_forecast_update,
)


@dataclass
class SampleState:
    """Rolling telemetry for one sample index."""

    target_class: int
    pred_class: int
    last_step: int
    history: deque[tuple[float, float, float, bool, int]] = field(default_factory=lambda: deque(maxlen=16))

    def push(self, epoch: float, p_true: float, margin: float, correct: bool, pred_class: int, step: int) -> None:
        """Append one telemetry observation to the rolling history.

        Example:
            >>> state.push(epoch=3.0, p_true=0.71, margin=0.22, correct=True, pred_class=1, step=120)
        """
        self.target_class = int(self.target_class)
        self.pred_class = int(pred_class)
        self.last_step = int(step)
        self.history.append((float(epoch), float(p_true), float(margin), bool(correct), int(pred_class)))


def _clamp01(x: float) -> float:
    """Clamp a numeric value into [0.0, 1.0].

    Example:
        >>> _clamp01(1.2)
        1.0
    """
    return max(0.0, min(1.0, x))


def _overlap(a: list[int], b: list[int]) -> float:
    """Return normalized overlap ratio between two candidate-id lists.

    Example:
        >>> _overlap([1, 2, 3], [2, 3, 4])
        0.6666666666666666
    """
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    return float(len(sa & sb) / float(max(1, min(len(sa), len(sb)))))


def _risk_score(state: SampleState) -> float | None:
    """Heuristic horizon risk score in [0, 1] from rolling telemetry."""
    n = len(state.history)
    if n < 3:
        return None
    p_true_vals = [h[1] for h in state.history]
    margin_vals = [h[2] for h in state.history]
    correct_vals = [1.0 if h[3] else 0.0 for h in state.history]
    pred_vals = [h[4] for h in state.history]

    p_true_mean = mean(p_true_vals)
    margin_mean = mean(margin_vals)
    err_rate = 1.0 - mean(correct_vals)
    flips = 0.0
    if n > 1:
        flips = sum(1 for i in range(1, n) if pred_vals[i] != pred_vals[i - 1]) / float(n - 1)
    trend = p_true_vals[-1] - p_true_vals[0]  # negative means worsening confidence
    stagnation = _clamp01((0.03 - trend) * 8.0)

    score = (
        (0.40 * err_rate)
        + (0.25 * (1.0 - p_true_mean))
        + (0.15 * (1.0 - _clamp01((margin_mean + 1.0) * 0.5)))
        + (0.15 * flips)
        + (0.05 * stagnation)
    )
    return _clamp01(score)


def _extract_epoch_accuracy(epoch_counts: dict[int, tuple[int, int]], min_samples: int = 32) -> list[tuple[int, float]]:
    """Convert epoch counters into `(epoch, accuracy)` rows.

    Example:
        >>> _extract_epoch_accuracy({2: (40, 30)}, min_samples=32)
        [(2, 0.75)]
    """
    rows: list[tuple[int, float]] = []
    for ep in sorted(epoch_counts):
        cnt, ok = epoch_counts[ep]
        if cnt >= min_samples:
            rows.append((ep, float(ok / max(1, cnt))))
    return rows


def main() -> None:
    """Run forecast worker loop and publish top-k likely failures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, default="http://localhost:8000")
    ap.add_argument("--backend", type=str, default="torch", choices=["torch", "tensorflow"])
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--horizon_epochs", type=int, default=10)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--warmup_epochs", type=float, default=5.0)
    ap.add_argument("--chance_acc", type=float, default=0.10)
    ap.add_argument("--competence_margin", type=float, default=0.15)
    ap.add_argument("--plateau_window", type=int, default=4)
    ap.add_argument("--plateau_delta", type=float, default=0.005)
    ap.add_argument("--min_error_rate", type=float, default=0.08)
    ap.add_argument("--stability_window", type=int, default=3)
    ap.add_argument("--stability_overlap", type=float, default=0.60)
    ap.add_argument("--cooldown_epochs", type=float, default=3.0)
    ap.add_argument("--enqueue_refresh_steps", type=int, default=400)
    ap.add_argument("--niceness", type=int, default=5)
    args = ap.parse_args()

    try:
        os.nice(args.niceness)
    except Exception:
        pass

    sample_states: dict[int, SampleState] = {}
    epoch_counts: dict[int, tuple[int, int]] = {}
    rank_history: deque[list[int]] = deque(maxlen=max(2, args.stability_window))
    last_feedback_sig = ""
    cooldown_until_epoch = 0.0
    window_open = False
    last_enqueued_step: dict[int, int] = {}
    last_post_step = -1

    post_event(
        args.server,
        "forecast_worker",
        {
            "status": "online",
            "backend": args.backend,
            "cpu_only": True,
            "horizon_epochs": int(args.horizon_epochs),
        },
    )

    while True:
        item = dequeue_forecast_telemetry(args.server, timeout_s=2.0, backend=args.backend)
        if item is None:
            time.sleep(args.poll)
            continue

        step = int(item.get("step", 0))
        epoch = float(item.get("epoch", 0.0))
        samples = item.get("samples", [])

        for s in samples:
            sample_idx = int(s["sample_idx"])
            target_class = int(s["target_class"])
            pred_class = int(s["pred_class"])
            p_true = float(s["p_true"])
            margin = float(s["margin"])
            correct = bool(s["correct"])

            state = sample_states.get(sample_idx)
            if state is None:
                state = SampleState(
                    target_class=target_class,
                    pred_class=pred_class,
                    last_step=step,
                )
                sample_states[sample_idx] = state
            state.target_class = target_class
            state.push(
                epoch=epoch,
                p_true=p_true,
                margin=margin,
                correct=correct,
                pred_class=pred_class,
                step=step,
            )

            epoch_key = int(epoch)
            cnt, ok = epoch_counts.get(epoch_key, (0, 0))
            epoch_counts[epoch_key] = (cnt + 1, ok + (1 if correct else 0))

        feedback = get_feedback(args.server)
        fb_sig = json.dumps(feedback, sort_keys=True, default=str) if feedback else ""
        if fb_sig and fb_sig != last_feedback_sig:
            last_feedback_sig = fb_sig
            has_intervention = bool(
                feedback.get("roi") is not None
                or feedback.get("roi_mask") is not None
                or str(feedback.get("instruction", "")).strip()
                or feedback.get("strength") is not None
            )
            if has_intervention:
                cooldown_until_epoch = max(cooldown_until_epoch, epoch + args.cooldown_epochs)

        epoch_acc_rows = _extract_epoch_accuracy(epoch_counts)
        recent_acc = epoch_acc_rows[-1][1] if epoch_acc_rows else 0.0
        warmup_pass = epoch >= args.warmup_epochs
        competence_pass = recent_acc >= (args.chance_acc + args.competence_margin)
        error_pass = (1.0 - recent_acc) >= args.min_error_rate

        plateau_pass = False
        if len(epoch_acc_rows) >= max(2, args.plateau_window + 1):
            vals = [v for _, v in epoch_acc_rows[-(args.plateau_window + 1) :]]
            improvement = vals[-1] - vals[0]
            plateau_pass = improvement <= args.plateau_delta

        ranked: list[tuple[int, float, SampleState]] = []
        for sample_idx, state in sample_states.items():
            risk = _risk_score(state)
            if risk is None:
                continue
            ranked.append((sample_idx, risk, state))
        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[: max(1, args.topk)]
        current_ids = [sid for sid, _, _ in ranked]

        stability_pass = False
        if len(rank_history) >= max(1, args.stability_window - 1):
            overlaps = [_overlap(current_ids, old) for old in list(rank_history)[-(args.stability_window - 1) :]]
            stability_pass = mean(overlaps) >= args.stability_overlap
        rank_history.append(current_ids)

        cooldown_pass = epoch >= cooldown_until_epoch
        all_pass = warmup_pass and competence_pass and plateau_pass and error_pass and stability_pass and cooldown_pass

        reason = "ready"
        if not warmup_pass:
            reason = "warmup"
        elif not competence_pass:
            reason = "competence"
        elif not plateau_pass:
            reason = "plateau"
        elif not error_pass:
            reason = "error_floor"
        elif not stability_pass:
            reason = "stability"
        elif not cooldown_pass:
            reason = "cooldown"

        candidates_payload = [
            {
                "sample_idx": int(sid),
                "risk_score": float(risk),
                "target_class": int(state.target_class),
                "pred_class": int(state.pred_class),
                "horizon_epochs": int(args.horizon_epochs),
            }
            for sid, risk, state in ranked
        ]

        should_post = (step != last_post_step) and (all_pass or step % 100 == 0)
        if should_post:
            last_post_step = step
            post_forecast_update(
                args.server,
                {
                    "step": step,
                    "epoch": epoch,
                    "window_open": bool(all_pass),
                    "reason": reason,
                    "candidates": candidates_payload,
                },
                backend=args.backend,
            )

        if all_pass:
            for cand in candidates_payload:
                sample_idx = int(cand["sample_idx"])
                last_sent = last_enqueued_step.get(sample_idx, -10**9)
                if (step - last_sent) < args.enqueue_refresh_steps:
                    continue
                last_enqueued_step[sample_idx] = step
                req = XaiRequestPayload(
                    step=step,
                    sample_idx=sample_idx,
                    target_class=int(cand["target_class"]),
                    pred_class=int(cand["pred_class"]),
                    request_kind="forecast",
                    risk_score=float(cand["risk_score"]),
                    horizon_epochs=int(args.horizon_epochs),
                    backend=args.backend,
                )
                enqueue_xai(args.server, req.to_dict(), backend=args.backend)

        if window_open != all_pass:
            window_open = all_pass
            post_event(
                args.server,
                "hitl_window",
                {
                    "backend": args.backend,
                    "step": step,
                    "epoch": epoch,
                    "window_open": bool(window_open),
                    "reason": reason,
                    "candidate_count": len(candidates_payload),
                },
            )


if __name__ == "__main__":
    main()

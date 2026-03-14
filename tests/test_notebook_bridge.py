from __future__ import annotations

from pathlib import Path

import coflect.modules.hilt.common.notebook_bridge as nb
from coflect.modules.hilt.common.notebook_bridge import NotebookBridgeConfig, NotebookHILTBridge


def test_roi_norm_to_pixels() -> None:
    pixels = nb.roi_norm_to_pixels((0.25, 0.25, 0.75, 0.75), height=64, width=64)
    assert pixels == (16, 16, 48, 48)


def test_maybe_post_metrics_cadence(monkeypatch) -> None:
    events: list[tuple[str, dict[str, float | int | str]]] = []

    def _fake_post_event(server: str, event_type: str, payload: dict[str, float | int | str]) -> None:
        events.append((event_type, payload))

    monkeypatch.setattr(nb, "post_event", _fake_post_event)

    bridge = NotebookHILTBridge(
        NotebookBridgeConfig(
            backend="torch",
            metrics_every=2,
            feedback_every=100,
        )
    )

    assert bridge.maybe_post_metrics(step=1, loss=1.0, acc=0.5) is False
    assert bridge.maybe_post_metrics(step=2, loss=0.9, acc=0.6) is True
    assert len(events) == 1
    event_type, payload = events[0]
    assert event_type == "metrics"
    assert payload["backend"] == "torch"
    assert payload["step"] == 2
    assert payload["loss"] == 0.9
    assert payload["acc"] == 0.6


def test_maybe_sync_feedback_applies_focus_pause_and_roi(monkeypatch) -> None:
    feedbacks = [
        {
            "step": 10,
            "sample_idx": 7,
            "instruction": "increase focus by 20% and center",
            "paused": True,
        },
        {
            "step": 10,
            "sample_idx": 7,
            "instruction": "increase focus by 20% and center",
            "paused": True,
        },
    ]

    def _fake_get_feedback(server: str) -> dict[str, object]:
        if feedbacks:
            return feedbacks.pop(0)
        return {}

    events: list[tuple[str, dict[str, object]]] = []

    def _fake_post_event(server: str, event_type: str, payload: dict[str, object]) -> None:
        events.append((event_type, payload))

    monkeypatch.setattr(nb, "get_feedback", _fake_get_feedback)
    monkeypatch.setattr(nb, "post_event", _fake_post_event)

    bridge = NotebookHILTBridge(
        NotebookBridgeConfig(
            backend="torch",
            metrics_every=100,
            feedback_every=1,
            initial_focus_lambda=0.0,
            emit_feedback_applied_event=True,
        )
    )

    update = bridge.maybe_sync_feedback(step=1)
    assert update.changed is True
    assert abs(update.focus_lambda - 0.2) < 1e-8
    assert update.paused is True
    assert update.roi_norm == (0.25, 0.25, 0.75, 0.75)
    assert update.sample_idx == 7
    assert bridge.focus_lambda == update.focus_lambda
    assert bridge.paused is True
    assert bridge.roi_norm == update.roi_norm
    assert any(et == "trainer_feedback_applied" for et, _ in events)

    # Same feedback should be ignored as no-change.
    update2 = bridge.maybe_sync_feedback(step=2)
    assert update2.changed is False


def test_maybe_enqueue_xai_cadence(monkeypatch) -> None:
    queued: list[dict[str, object]] = []
    events: list[tuple[str, dict[str, object]]] = []

    def _fake_enqueue_xai(server: str, payload: dict[str, object], backend: str = "torch") -> None:
        p = dict(payload)
        p["backend_key"] = backend
        queued.append(p)

    def _fake_post_event(server: str, event_type: str, payload: dict[str, object]) -> None:
        events.append((event_type, payload))

    monkeypatch.setattr(nb, "enqueue_xai", _fake_enqueue_xai)
    monkeypatch.setattr(nb, "post_event", _fake_post_event)

    bridge = NotebookHILTBridge(
        NotebookBridgeConfig(
            backend="torch",
            metrics_every=100,
            feedback_every=100,
            xai_every=3,
            emit_xai_requested_event=True,
        )
    )

    assert (
        bridge.maybe_enqueue_xai(
            step=1,
            sample_idx=10,
            target_class=0,
            pred_class=1,
        )
        is False
    )
    assert (
        bridge.maybe_enqueue_xai(
            step=3,
            sample_idx=10,
            target_class=0,
            pred_class=1,
        )
        is True
    )

    assert len(queued) == 1
    payload = queued[0]
    assert payload["step"] == 3
    assert payload["sample_idx"] == 10
    assert payload["target_class"] == 0
    assert payload["pred_class"] == 1
    assert payload["backend"] == "torch"
    assert payload["backend_key"] == "torch"
    assert any(et == "xai_requested" for et, _ in events)


def test_xai_worker_script_fallback(tmp_path: Path) -> None:
    missing_script = str(tmp_path / "missing_worker.py")
    bridge_missing = NotebookHILTBridge(
        NotebookBridgeConfig(
            backend="tensorflow",
            xai_worker_script=missing_script,
        )
    )
    cmd_missing = bridge_missing._build_xai_cmd()
    assert "-m" in cmd_missing
    assert "coflect.modules.hilt.xai_worker.worker_tf_livecam" in cmd_missing

    existing_script = tmp_path / "worker.py"
    existing_script.write_text("print('ok')", encoding="utf-8")
    bridge_existing = NotebookHILTBridge(
        NotebookBridgeConfig(
            backend="tensorflow",
            xai_worker_script=str(existing_script),
        )
    )
    cmd_existing = bridge_existing._build_xai_cmd()
    assert "-m" not in cmd_existing
    assert str(existing_script) in cmd_existing


def test_xai_worker_script_resolution_from_examples_path(tmp_path: Path, monkeypatch) -> None:
    examples_dir = tmp_path / "examples" / "hilt"
    examples_dir.mkdir(parents=True)
    script_path = examples_dir / "coflect_tf_xai_worker_local.py"
    script_path.write_text("print('ok')", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    bridge = NotebookHILTBridge(
        NotebookBridgeConfig(
            backend="tensorflow",
            xai_worker_script="coflect_tf_xai_worker_local.py",
        )
    )
    cmd = bridge._build_xai_cmd()
    assert "-m" not in cmd
    assert str(script_path.resolve()) in cmd

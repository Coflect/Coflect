"""Notebook-friendly bridge for live HILT UI integration.

This module keeps notebook training loops simple while allowing:
- lightweight metric streaming to the backend/UI
- deterministic text/ROI feedback polling from the backend

The heavy XAI path still stays out-of-process; this bridge only sends compact
JSON payloads through the existing backend wire protocol.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO
from urllib.request import Request, urlopen

from coflect.modules.hilt.common.instruction_parser import ParsedInstruction, parse_instruction
from coflect.modules.hilt.common.messages import XaiRequestPayload
from coflect.modules.hilt.common.wire import enqueue_xai, get_feedback, post_event

JSONDict = dict[str, Any]
_MANAGED_PROCS: dict[str, subprocess.Popen[bytes]] = {}
_MANAGED_LOGS: dict[str, BinaryIO] = {}


def _clamp01(x: float) -> float:
    """Clamp a numeric value into [0.0, 1.0]."""
    return max(0.0, min(1.0, x))


def _parse_sample_idx(raw: Any) -> int | None:
    """Parse optional sample index from feedback payload."""
    if isinstance(raw, int) and raw >= 0:
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None


def _parse_roi_norm(raw: Any) -> tuple[float, float, float, float] | None:
    """Parse normalized ROI `(x0,y0,x1,y1)` from feedback `roi` payload.

    Accepted formats:
    - dict with keys `x0,y0,x1,y1`
    - list/tuple of four numeric values
    """
    values: list[Any]
    if isinstance(raw, dict):
        values = [raw.get("x0"), raw.get("y0"), raw.get("x1"), raw.get("y1")]
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = list(raw)
    else:
        return None

    if any(v is None for v in values):
        return None

    try:
        x0, y0, x1, y1 = (float(v) for v in values)
    except (TypeError, ValueError):
        return None

    # Notebook/UI ROI contract is normalized coordinates in [0, 1].
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.5:
        return None

    x0c, x1c = sorted((_clamp01(x0), _clamp01(x1)))
    y0c, y1c = sorted((_clamp01(y0), _clamp01(y1)))
    if x1c <= x0c or y1c <= y0c:
        return None
    return (x0c, y0c, x1c, y1c)


def roi_norm_to_pixels(
    roi_norm: tuple[float, float, float, float],
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    """Convert normalized ROI into bounded pixel coordinates.

    Example:
        >>> roi_norm_to_pixels((0.25, 0.25, 0.75, 0.75), height=64, width=64)
        (16, 16, 48, 48)
    """
    if height <= 0 or width <= 0:
        return None
    x0n, y0n, x1n, y1n = roi_norm
    x0 = int(max(0, min(width - 1, min(x0n, x1n) * width)))
    x1 = int(max(1, min(width, max(x0n, x1n) * width)))
    y0 = int(max(0, min(height - 1, min(y0n, y1n) * height)))
    y1 = int(max(1, min(height, max(y0n, y1n) * height)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


@dataclass(frozen=True)
class NotebookBridgeConfig:
    """Configuration for notebook-to-HILT bridge behavior."""

    server: str = "http://127.0.0.1:8000"
    backend: str = "torch"
    metrics_every: int = 10
    feedback_every: int = 10
    xai_every: int = 100
    auto_start_backend: bool = False
    auto_start_xai_worker: bool = False
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    startup_timeout_s: float = 20.0
    startup_poll_s: float = 0.5
    log_dir: str = "./.coflect_logs/notebook_bridge"
    xai_snapshot_dir: str = "./snapshots"
    xai_num_classes: int | None = None
    xai_image_size: int | None = None
    xai_dataset: str = "cifar10_catsdogs"
    data_root: str = "./data"
    split: str = "train"
    download_data: bool = True
    xai_method: str = "consensus"
    xai_device: str = ""
    xai_worker_module_override: str = ""
    xai_worker_script: str = ""
    initial_focus_lambda: float = 0.0
    initial_paused: bool = False
    emit_feedback_applied_event: bool = True
    emit_xai_requested_event: bool = True


@dataclass(frozen=True)
class FeedbackUpdate:
    """Structured feedback state update consumed by notebook training loops."""

    changed: bool
    focus_lambda: float
    paused: bool
    roi_norm: tuple[float, float, float, float] | None
    instruction: str
    parsed: ParsedInstruction
    sample_idx: int | None
    raw: JSONDict = field(default_factory=dict)


class NotebookHILTBridge:
    """Small stateful bridge for streaming metrics and consuming feedback.

    Example:
        >>> bridge = NotebookHILTBridge(NotebookBridgeConfig(backend="torch"))
        >>> _ = bridge.maybe_post_metrics(step=10, loss=0.42, acc=0.81)
    """

    def __init__(self, config: NotebookBridgeConfig):
        """Initialize bridge state from config defaults."""
        self.config = config
        self.metrics_every = max(1, int(config.metrics_every))
        self.feedback_every = max(1, int(config.feedback_every))
        self.xai_every = max(1, int(config.xai_every))
        self.focus_lambda = _clamp01(float(config.initial_focus_lambda))
        self.paused = bool(config.initial_paused)
        self.roi_norm: tuple[float, float, float, float] | None = None

        self._last_feedback: JSONDict = {}
        self._last_metric_step = 0
        self._last_metric_time = time.time()
        self._ensure_services()

    def _ensure_services(self) -> None:
        """Optionally auto-start backend and XAI worker for notebook-only flow."""
        server_ready = self._is_server_ready(timeout_s=0.5)
        if self.config.auto_start_backend and not server_ready:
            backend_key = f"backend:{self.config.server}"
            if not self._process_alive(backend_key):
                cmd = [
                    sys.executable,
                    "-m",
                    "coflect.modules.hilt.backend.app",
                    "--host",
                    self.config.backend_host,
                    "--port",
                    str(self.config.backend_port),
                ]
                self._launch_managed(backend_key, cmd)
            server_ready = self._wait_server_ready()

        if not self.config.auto_start_xai_worker or not server_ready:
            return

        xai_key = f"xai:{self.config.backend}:{self.config.server}"
        if self._process_alive(xai_key):
            return

        cmd = self._build_xai_cmd()
        if cmd:
            self._launch_managed(xai_key, cmd)

    def _build_xai_cmd(self) -> list[str]:
        """Build backend-specific XAI worker command for auto-start path."""
        def _prefix(default_module: str) -> list[str]:
            script = self.config.xai_worker_script.strip()
            if script:
                script_path = self._resolve_worker_script(script)
                if script_path is not None:
                    return [sys.executable, str(script_path)]
            module = self.config.xai_worker_module_override.strip() or default_module
            return [sys.executable, "-m", module]

        backend = self.config.backend.strip().lower()
        if backend == "torch":
            cmd = _prefix("coflect.modules.hilt.xai_worker.worker_torch_livecam") + [
                "--server",
                self.config.server,
                "--dataset",
                self.config.xai_dataset,
                "--data_root",
                self.config.data_root,
                "--split",
                self.config.split,
                "--snapshot_dir",
                self.config.xai_snapshot_dir,
                "--xai_method",
                self.config.xai_method,
            ]
            if self.config.download_data:
                cmd.append("--download_data")
            if self.config.xai_device.strip():
                cmd.extend(["--device", self.config.xai_device.strip()])
            return cmd

        if backend == "tensorflow":
            cmd = _prefix("coflect.modules.hilt.xai_worker.worker_tf_livecam") + [
                "--server",
                self.config.server,
                "--dataset",
                self.config.xai_dataset,
                "--data_root",
                self.config.data_root,
                "--split",
                self.config.split,
                "--snapshot_dir",
                self.config.xai_snapshot_dir,
                "--xai_method",
                self.config.xai_method,
            ]
            if self.config.download_data:
                cmd.append("--download_data")
            if self.config.xai_num_classes is not None and int(self.config.xai_num_classes) > 0:
                cmd.extend(["--num_classes", str(int(self.config.xai_num_classes))])
            if self.config.xai_image_size is not None and int(self.config.xai_image_size) > 0:
                cmd.extend(["--image_size", str(int(self.config.xai_image_size))])
            if self.config.xai_device.strip():
                cmd.extend(["--device", self.config.xai_device.strip()])
            return cmd

        return []

    def _resolve_worker_script(self, script: str) -> Path | None:
        """Resolve worker script from common notebook/repo-relative locations."""
        raw = Path(script).expanduser()
        candidates: list[Path] = [raw]
        if not raw.is_absolute():
            cwd = Path.cwd()
            name = raw.name
            candidates.extend(
                [
                    cwd / raw,
                    cwd / name,
                    cwd / "examples" / "hilt" / name,
                    cwd / "HILTM" / "examples" / "hilt" / name,
                ]
            )
            try:
                repo_root = Path(__file__).resolve().parents[4]
            except Exception:
                repo_root = None
            if repo_root is not None:
                candidates.extend(
                    [
                        repo_root / raw,
                        repo_root / "examples" / "hilt" / name,
                        repo_root / "HILTM" / "examples" / "hilt" / name,
                    ]
                )
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved.exists() and resolved.is_file():
                return resolved
        return None

    def _is_server_ready(self, timeout_s: float) -> bool:
        """Return True when backend root responds over HTTP."""
        try:
            req = Request(f"{self.config.server}/", method="GET")
            with urlopen(req, timeout=timeout_s):
                return True
        except Exception:
            return False

    def _wait_server_ready(self) -> bool:
        """Wait until backend is reachable or timeout expires."""
        deadline = time.time() + max(0.5, float(self.config.startup_timeout_s))
        poll_s = max(0.1, float(self.config.startup_poll_s))
        while time.time() < deadline:
            if self._is_server_ready(timeout_s=poll_s):
                return True
            time.sleep(poll_s)
        return False

    def _launch_managed(self, key: str, cmd: list[str]) -> None:
        """Launch a managed sidecar process with dedicated log file."""
        log_dir = Path(self.config.log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_key = key.replace(":", "_").replace("/", "_")
        log_path = log_dir / f"{safe_key}.log"
        log_handle = log_path.open("ab")
        try:
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
        except Exception:
            log_handle.close()
            return
        _MANAGED_PROCS[key] = proc
        _MANAGED_LOGS[key] = log_handle

    def _process_alive(self, key: str) -> bool:
        """Check if a managed process key is alive and clean stale entries."""
        proc = _MANAGED_PROCS.get(key)
        if proc is None:
            return False
        if proc.poll() is None:
            return True
        _MANAGED_PROCS.pop(key, None)
        log = _MANAGED_LOGS.pop(key, None)
        if log is not None:
            log.close()
        return False

    def maybe_post_metrics(
        self,
        *,
        step: int,
        loss: float,
        acc: float | None = None,
        epoch: float | None = None,
        focus_lambda: float | None = None,
        extra: JSONDict | None = None,
        force: bool = False,
    ) -> bool:
        """Post a compact `metrics` event at configured cadence.

        Returns `True` when an event is emitted.
        """
        if step < 0:
            return False
        if not force and step % self.metrics_every != 0:
            return False

        now = time.time()
        dt = max(1e-9, now - self._last_metric_time)
        ds = max(0, step - self._last_metric_step)
        sps = float(ds) / dt if ds > 0 else 0.0
        self._last_metric_time = now
        self._last_metric_step = step

        focus = self.focus_lambda if focus_lambda is None else _clamp01(float(focus_lambda))
        payload: JSONDict = {
            "backend": self.config.backend,
            "step": int(step),
            "loss": float(loss),
            "focus_lambda": float(focus),
            "sps": float(sps),
        }
        if acc is not None:
            payload["acc"] = float(acc)
        if epoch is not None:
            payload["epoch"] = float(epoch)
        if extra:
            payload.update(extra)

        post_event(self.config.server, "metrics", payload)
        return True

    def maybe_enqueue_xai(
        self,
        *,
        step: int,
        sample_idx: int,
        target_class: int,
        pred_class: int,
        request_kind: str = "periodic",
        risk_score: float | None = None,
        horizon_epochs: int | None = None,
        force: bool = False,
    ) -> bool:
        """Enqueue lightweight XAI request at configured cadence.

        Returns `True` when a request is emitted.
        """
        # Keep service state resilient: if sidecars died or were not started at
        # bridge construction time, retry auto-start on enqueue boundaries.
        self._ensure_services()

        if step < 0 or sample_idx < 0 or target_class < 0 or pred_class < 0:
            return False
        if not force and step % self.xai_every != 0:
            return False

        payload = XaiRequestPayload(
            step=int(step),
            sample_idx=int(sample_idx),
            target_class=int(target_class),
            pred_class=int(pred_class),
            request_kind=str(request_kind),
            risk_score=float(risk_score) if risk_score is not None else None,
            horizon_epochs=int(horizon_epochs) if horizon_epochs is not None else None,
            backend=self.config.backend,
        ).to_dict()
        enqueue_xai(self.config.server, payload, backend=self.config.backend)
        if self.config.emit_xai_requested_event:
            post_event(self.config.server, "xai_requested", payload)
        return True

    def maybe_sync_feedback(self, *, step: int, force: bool = False) -> FeedbackUpdate:
        """Poll and apply UI feedback at configured cadence.

        Returns a `FeedbackUpdate`. If `changed=False`, internal state was not
        modified.
        """
        if step < 0:
            return self._no_change(raw={})
        if not force and step % self.feedback_every != 0:
            return self._no_change(raw={})

        raw = get_feedback(self.config.server)
        if not isinstance(raw, dict):
            raw = {}

        if not raw or raw == self._last_feedback:
            return self._no_change(raw=raw)

        self._last_feedback = dict(raw)
        instruction = str(raw.get("instruction", ""))
        parsed = parse_instruction(instruction)

        raw_strength = raw.get("strength", None)
        try:
            base_strength = float(raw_strength) if raw_strength is not None else self.focus_lambda
        except (TypeError, ValueError):
            base_strength = self.focus_lambda

        next_focus = _clamp01(base_strength + parsed.strength_delta)
        if parsed.strength is not None:
            next_focus = parsed.strength

        next_paused = self.paused
        if raw.get("paused", None) is not None:
            next_paused = bool(raw["paused"])

        roi_updated = bool(raw.get("roi", None) is not None or parsed.roi_norm is not None)
        next_roi = self.roi_norm
        if roi_updated:
            next_roi = _parse_roi_norm(raw.get("roi"))
            if next_roi is None and parsed.roi_norm is not None:
                next_roi = parsed.roi_norm

        self.focus_lambda = next_focus
        self.paused = next_paused
        self.roi_norm = next_roi
        sample_idx = _parse_sample_idx(raw.get("sample_idx"))

        if self.config.emit_feedback_applied_event:
            post_event(
                self.config.server,
                "trainer_feedback_applied",
                {
                    "step": step,
                    "backend": self.config.backend,
                    "paused": self.paused,
                    "new_focus_lambda": self.focus_lambda,
                    "roi": self.roi_norm,
                    "sample_idx": sample_idx,
                    "instruction_policy": parsed.to_dict(),
                },
            )

        return FeedbackUpdate(
            changed=True,
            focus_lambda=self.focus_lambda,
            paused=self.paused,
            roi_norm=self.roi_norm,
            instruction=instruction,
            parsed=parsed,
            sample_idx=sample_idx,
            raw=raw,
        )

    def _no_change(self, raw: JSONDict) -> FeedbackUpdate:
        """Return immutable no-change feedback result."""
        return FeedbackUpdate(
            changed=False,
            focus_lambda=self.focus_lambda,
            paused=self.paused,
            roi_norm=self.roi_norm,
            instruction="",
            parsed=ParsedInstruction(),
            sample_idx=None,
            raw=raw,
        )

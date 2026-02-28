"""HTTP wire helpers for trainer/worker <-> backend communication.

These helpers are intentionally fail-soft: transient network failures should
not crash the trainer hot path.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

requests: Any

try:
    requests = import_module("requests")

    _REQUEST_EXCEPTION: type[Exception] = requests.RequestException
except Exception:  # pragma: no cover - optional dependency guard
    requests = None
    _REQUEST_EXCEPTION = Exception

JSONDict = dict[str, Any]

_LOG = logging.getLogger(__name__)
_SHORT_TIMEOUT_S = 5.0
_POLL_TIMEOUT_S = 10.0


def post_event(server: str, event_type: str, payload: JSONDict) -> None:
    """Post a fire-and-forget event to backend."""
    if requests is None:
        return
    try:
        requests.post(
            f"{server}/event",
            json={"type": event_type, "payload": payload},
            timeout=_SHORT_TIMEOUT_S,
        )
    except _REQUEST_EXCEPTION as exc:
        _LOG.debug("post_event failed: %s", exc)


def get_feedback(server: str) -> JSONDict:
    """Fetch latest feedback payload from backend."""
    if requests is None:
        return {}
    try:
        response = requests.get(f"{server}/feedback", timeout=_SHORT_TIMEOUT_S)
    except _REQUEST_EXCEPTION as exc:
        _LOG.debug("get_feedback failed: %s", exc)
        return {}
    return response.json().get("feedback", {}) if response.ok else {}


def enqueue_xai(server: str, payload: JSONDict, backend: str = "torch") -> None:
    """Enqueue an asynchronous XAI request."""
    if requests is None:
        return
    body = dict(payload)
    body.setdefault("backend", backend)
    try:
        requests.post(f"{server}/xai/request", json=body, timeout=_SHORT_TIMEOUT_S)
    except _REQUEST_EXCEPTION as exc:
        _LOG.debug("enqueue_xai failed: %s", exc)


def dequeue_xai(server: str, timeout_s: float = 2.0, backend: str = "torch") -> JSONDict | None:
    """Dequeue an XAI request or return None on timeout/error."""
    if requests is None:
        return None
    try:
        response = requests.get(
            f"{server}/xai/next",
            params={"timeout_s": timeout_s, "backend": backend},
            timeout=_POLL_TIMEOUT_S,
        )
    except _REQUEST_EXCEPTION as exc:
        _LOG.debug("dequeue_xai failed: %s", exc)
        return None
    if not response.ok:
        return None
    return response.json().get("item", None)


def enqueue_forecast_telemetry(server: str, payload: JSONDict, backend: str = "torch") -> None:
    """Enqueue compact telemetry for CPU-only forecast worker."""
    if requests is None:
        return
    body = dict(payload)
    body.setdefault("backend", backend)
    try:
        requests.post(f"{server}/forecast/telemetry", json=body, timeout=_SHORT_TIMEOUT_S)
    except _REQUEST_EXCEPTION as exc:
        _LOG.debug("enqueue_forecast_telemetry failed: %s", exc)


def dequeue_forecast_telemetry(server: str, timeout_s: float = 2.0, backend: str = "torch") -> JSONDict | None:
    """Dequeue telemetry batch for forecast worker or return None on timeout/error."""
    if requests is None:
        return None
    try:
        response = requests.get(
            f"{server}/forecast/next",
            params={"timeout_s": timeout_s, "backend": backend},
            timeout=_POLL_TIMEOUT_S,
        )
    except _REQUEST_EXCEPTION as exc:
        _LOG.debug("dequeue_forecast_telemetry failed: %s", exc)
        return None
    if not response.ok:
        return None
    return response.json().get("item", None)


def post_forecast_update(server: str, payload: JSONDict, backend: str = "torch") -> None:
    """Publish forecast top-k state for UI consumption."""
    if requests is None:
        return
    body = dict(payload)
    body.setdefault("backend", backend)
    try:
        requests.post(f"{server}/forecast/update", json=body, timeout=_SHORT_TIMEOUT_S)
    except _REQUEST_EXCEPTION as exc:
        _LOG.debug("post_forecast_update failed: %s", exc)


def get_latest_forecast(server: str, backend: str = "torch") -> JSONDict:
    """Fetch latest forecast state from backend."""
    if requests is None:
        return {}
    try:
        response = requests.get(
            f"{server}/forecast/latest",
            params={"backend": backend},
            timeout=_SHORT_TIMEOUT_S,
        )
    except _REQUEST_EXCEPTION as exc:
        _LOG.debug("get_latest_forecast failed: %s", exc)
        return {}
    if not response.ok:
        return {}
    return response.json().get("forecast", {})

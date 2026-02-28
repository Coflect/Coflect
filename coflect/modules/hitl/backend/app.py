"""FastAPI backend for event ingest, WS broadcast, feedback, XAI, and forecast queues."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="Coflect HITL Event Server")
_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Allow local browser clients.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ----------------------------
# In-memory state (minimal)
# ----------------------------

@dataclass
class State:
    clients: set[WebSocket] = field(default_factory=set)
    latest_feedback: dict[str, Any] = field(default_factory=dict)
    xai_queues: dict[str, asyncio.Queue[dict[str, Any]]] = field(default_factory=dict)
    forecast_queues: dict[str, asyncio.Queue[dict[str, Any]]] = field(default_factory=dict)
    forecast_latest: dict[str, dict[str, Any]] = field(default_factory=dict)

STATE = State()

# ----------------------------
# Wire models
# ----------------------------

class Event(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


RoiBox = dict[str, float] | list[float]
RoiMask = list[float] | list[list[float]]


class Feedback(BaseModel):
    step: int = Field(ge=0)
    sample_idx: int = Field(ge=0)
    instruction: str = ""
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    paused: bool | None = None
    roi: RoiBox | None = None
    roi_mask: RoiMask | None = None


class XaiRequest(BaseModel):
    step: int = Field(ge=0)
    sample_idx: int = Field(ge=0)
    target_class: int = Field(ge=0)
    pred_class: int = Field(ge=0)
    request_kind: str = Field(default="periodic", min_length=1)
    risk_score: float | None = None
    horizon_epochs: int | None = Field(default=None, ge=1)
    backend: str = Field(default="torch", min_length=1)


class ForecastTelemetrySample(BaseModel):
    sample_idx: int = Field(ge=0)
    target_class: int = Field(ge=0)
    pred_class: int = Field(ge=0)
    p_true: float = Field(ge=0.0, le=1.0)
    margin: float = Field(ge=-1.0, le=1.0)
    correct: bool


class ForecastTelemetryBatch(BaseModel):
    step: int = Field(ge=0)
    epoch: float = Field(ge=0.0)
    backend: str = Field(default="torch", min_length=1)
    samples: list[ForecastTelemetrySample] = Field(default_factory=list)


class ForecastCandidate(BaseModel):
    sample_idx: int = Field(ge=0)
    risk_score: float = Field(ge=0.0, le=1.0)
    target_class: int = Field(ge=0)
    pred_class: int = Field(ge=0)
    horizon_epochs: int = Field(ge=1)


class ForecastUpdate(BaseModel):
    backend: str = Field(default="torch", min_length=1)
    step: int = Field(ge=0)
    epoch: float = Field(ge=0.0)
    window_open: bool = False
    reason: str = ""
    candidates: list[ForecastCandidate] = Field(default_factory=list)


@app.get("/", response_model=None)
async def ui_index() -> FileResponse:
    """Serve the built-in static HITL dashboard without Node tooling."""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI static assets are missing.")
    return FileResponse(index_path)

# ----------------------------
# WebSocket: UI subscribes here
# ----------------------------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """Attach a UI websocket client and keep it subscribed until disconnect."""
    await ws.accept()
    STATE.clients.add(ws)
    try:
        while True:
            # UI can optionally send pings; we ignore payload
            await ws.receive_text()
    except WebSocketDisconnect:
        STATE.clients.discard(ws)


async def broadcast(event: Event) -> None:
    """Broadcast event to all connected UI clients."""
    dead: list[WebSocket] = []
    msg = event.model_dump_json()
    for ws in STATE.clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.clients.discard(ws)

# ----------------------------
# Trainer/XAI worker post events here
# ----------------------------

@app.post("/event")
async def post_event(event: Event) -> dict[str, Any]:
    """Ingest event from trainer/worker and fan out over websocket."""
    await broadcast(event)
    return {"ok": True, "clients": len(STATE.clients)}

# ----------------------------
# Feedback: UI -> server, trainer polls
# ----------------------------

@app.post("/feedback")
async def post_feedback(feedback: Feedback) -> dict[str, Any]:
    """Store latest feedback and broadcast update to subscribers."""
    STATE.latest_feedback = feedback.model_dump()
    await broadcast(Event(type="feedback", payload=STATE.latest_feedback))
    return {"ok": True}


@app.get("/feedback")
async def get_feedback() -> dict[str, Any]:
    """Return the latest feedback payload for trainer polling."""
    return {"ok": True, "feedback": STATE.latest_feedback}

# ----------------------------
# XAI queue: trainer enqueues, worker dequeues
# ----------------------------

def _get_xai_queue(backend: str) -> asyncio.Queue[dict[str, Any]]:
    """Return/create the XAI queue for a backend key.

    Example:
        >>> q = _get_xai_queue("torch")
    """
    key = backend.strip().lower() or "torch"
    if key not in STATE.xai_queues:
        STATE.xai_queues[key] = asyncio.Queue()
    return STATE.xai_queues[key]


def _get_forecast_queue(backend: str) -> asyncio.Queue[dict[str, Any]]:
    """Return/create bounded forecast queue for a backend key.

    Example:
        >>> q = _get_forecast_queue("torch")
    """
    key = backend.strip().lower() or "torch"
    if key not in STATE.forecast_queues:
        STATE.forecast_queues[key] = asyncio.Queue(maxsize=4096)
    return STATE.forecast_queues[key]


@app.post("/xai/request")
async def enqueue_xai(req: XaiRequest) -> dict[str, Any]:
    """Enqueue an XAI task from trainer."""
    queue = _get_xai_queue(req.backend)
    await queue.put(req.model_dump())
    return {"ok": True, "queued": queue.qsize(), "backend": req.backend}


@app.get("/xai/next")
async def dequeue_xai(timeout_s: float = 2.0, backend: str = "torch") -> dict[str, Any]:
    """Dequeue the next XAI task, returning `item: None` on timeout."""
    queue = _get_xai_queue(backend)
    try:
        item = await asyncio.wait_for(queue.get(), timeout=timeout_s)
        return {"ok": True, "item": item, "backend": backend}
    except asyncio.TimeoutError:
        return {"ok": True, "item": None, "backend": backend}


# ----------------------------
# Forecast queue: trainer enqueues telemetry, forecast worker dequeues
# ----------------------------

@app.post("/forecast/telemetry")
async def enqueue_forecast_telemetry(batch: ForecastTelemetryBatch) -> dict[str, Any]:
    """Enqueue compact telemetry samples for CPU-only forecast worker."""
    queue = _get_forecast_queue(batch.backend)
    item = batch.model_dump()
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        return {"ok": False, "dropped": True, "backend": batch.backend}
    return {"ok": True, "queued": queue.qsize(), "backend": batch.backend}


@app.get("/forecast/next")
async def dequeue_forecast_telemetry(timeout_s: float = 2.0, backend: str = "torch") -> dict[str, Any]:
    """Dequeue forecast telemetry, returning `item: None` on timeout."""
    queue = _get_forecast_queue(backend)
    try:
        item = await asyncio.wait_for(queue.get(), timeout=timeout_s)
        return {"ok": True, "item": item, "backend": backend}
    except asyncio.TimeoutError:
        return {"ok": True, "item": None, "backend": backend}


@app.post("/forecast/update")
async def post_forecast_update(update: ForecastUpdate) -> dict[str, Any]:
    """Store latest forecast ranking and broadcast to UI."""
    key = update.backend.strip().lower() or "torch"
    payload = update.model_dump()
    STATE.forecast_latest[key] = payload
    await broadcast(Event(type="forecast_topk", payload=payload))
    return {"ok": True, "backend": key}


@app.get("/forecast/latest")
async def get_forecast_latest(backend: str = "torch") -> dict[str, Any]:
    """Return most recent forecast state for a backend."""
    key = backend.strip().lower() or "torch"
    return {"ok": True, "backend": key, "forecast": STATE.forecast_latest.get(key, {})}


def main() -> None:
    """Run backend server with uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required; install with `pip install coflect[server]`") from exc

    parser = argparse.ArgumentParser(description="Run Coflect HITL backend server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("coflect.modules.hitl.backend.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

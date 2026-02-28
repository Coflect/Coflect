#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DURATION="${1:-20}"
BACKEND="${2:-torch}"
DATASET="${3:-synthetic}"
SERVER="http://127.0.0.1:8000"

BACKEND_LOG="/tmp/coflect_backend.log"
TRAINER_LOG="/tmp/coflect_trainer.log"
WORKER_LOG="/tmp/coflect_worker.log"
FORECAST_LOG="/tmp/coflect_forecast.log"

cleanup() {
  set +e
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${TRAINER_PID:-}" ]] && kill "$TRAINER_PID" 2>/dev/null || true
  [[ -n "${FORECAST_PID:-}" ]] && kill "$FORECAST_PID" 2>/dev/null || true
  [[ -n "${WORKER_PID:-}" ]] && kill "$WORKER_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$TRAINER_PID" "$FORECAST_PID" "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

coflect-hitl-backend --host 127.0.0.1 --port 8000 >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
sleep 2

if [[ "$BACKEND" == "tensorflow" ]]; then
  if [[ "$DATASET" != "synthetic" ]]; then
    echo "Note: TensorFlow smoke currently uses synthetic dataset only; ignoring DATASET=$DATASET" >&2
  fi
  TRAINER_CMD=(coflect-hitl-trainer-tf --server "$SERVER" --steps 200 --xai_every 50 --snapshot_every 100)
  FORECAST_CMD=(coflect-hitl-forecast-worker --server "$SERVER" --backend tensorflow)
  WORKER_CMD=(coflect-hitl-xai-worker-tf --server "$SERVER" --xai_method consensus)
elif [[ "$BACKEND" == "torch" ]]; then
  TRAINER_CMD=(coflect-hitl-trainer-torch --server "$SERVER" --steps 200 --xai_every 50 --snapshot_every 100 --dataset "$DATASET")
  FORECAST_CMD=(coflect-hitl-forecast-worker --server "$SERVER" --backend torch)
  WORKER_CMD=(coflect-hitl-xai-worker-torch --server "$SERVER" --xai_method consensus --dataset "$DATASET")
  if [[ "$DATASET" == "cifar10_catsdogs" ]]; then
    TRAINER_CMD+=(--data_root ./data --download_data)
    WORKER_CMD+=(--data_root ./data --download_data)
  fi
else
  echo "Unsupported backend: $BACKEND (expected: torch|tensorflow)" >&2
  exit 2
fi

"${TRAINER_CMD[@]}" >"$TRAINER_LOG" 2>&1 &
TRAINER_PID=$!

"${FORECAST_CMD[@]}" >"$FORECAST_LOG" 2>&1 &
FORECAST_PID=$!

"${WORKER_CMD[@]}" >"$WORKER_LOG" 2>&1 &
WORKER_PID=$!

sleep "$DURATION"

curl -sf "$SERVER/feedback" >/dev/null

echo "Smoke check completed. Logs:"
echo "- $BACKEND_LOG"
echo "- $TRAINER_LOG"
echo "- $FORECAST_LOG"
echo "- $WORKER_LOG"

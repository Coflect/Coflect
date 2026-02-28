# Changelog

## Unreleased

- Added TensorFlow/Keras HITL runtime entrypoints:
  - `coflect-hitl-trainer-tf`
  - `coflect-hitl-xai-worker-tf`
- Added TensorFlow trainer loop with async snapshot writes and backend-specific XAI queue requests.
- Added TensorFlow XAI worker with snapshot sync, attribution timing, and XAI budget controls.
- Added backend-specific XAI queueing in backend API (`backend` routing for request/dequeue).
- Added flexible deterministic instruction parsing and optional feedback `strength` override.
- Added richer XAI payload telemetry (`top_classes`, `top_probs`, `xai_agreement`).
- Added LiveCAM naming (with backward-compatible Grad-CAM module wrappers).
- Added pause/resume feedback control for trainers.
- Added mistake-focused XAI request path (`request_kind="mistake"`) and UI queue for up to 10 misclassified examples.
- Added CPU-only forecast worker (`coflect-hitl-forecast-worker`) that ranks top-k likely future failures from compact trainer telemetry.
- Added backend forecast queue API (`/forecast/telemetry`, `/forecast/next`, `/forecast/update`, `/forecast/latest`).
- Added review window gating (warmup, competence, plateau, error-floor, stability, cooldown) and `forecast_topk`/`hitl_window` events.
- Updated UI to consume forecast candidates and support ROI feedback on top forecasted failures.
- Added `modality_focus` field in XAI payloads (currently `{"image": 1.0}`) for multimodal-ready protocol evolution.
- Replaced Node/Vite UI with static HTML/CSS/JS served by FastAPI at `/` (no npm runtime dependency).

## 0.1.0 - 2026-02-22

- Packaged project as `coflect` for PyPI.
- Added pluggable backend and module registry foundations.
- Moved HITL module under `coflect.modules.hitl` namespace.
- Added CLI entrypoints:
  - `coflect-hitl-backend`
  - `coflect-hitl-trainer-torch`
  - `coflect-hitl-xai-worker-torch`
- Added CI and PyPI publish workflows.
- Added support matrix, release playbook, and contribution guide.

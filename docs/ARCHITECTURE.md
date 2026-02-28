# Architecture

## Package Layout

- `coflect/backends/`
  - backend abstraction and framework-specific implementations
  - registry for backend factories
- `coflect/modules/`
  - module abstraction and registry
  - `hitl/` module implementation

## HITL Runtime Topology

- `backend` process: event ingest + websocket broadcast + feedback storage + backend-specific XAI/forecast queues.
- `trainer` process: emits scalar metrics and compact telemetry, enqueues lightweight XAI requests.
- `forecast_worker` process: CPU-only ranking of likely future failures and HITL window gating.
- `xai_worker` process: loads snapshots asynchronously, computes LiveCAM attribution, publishes compressed image payloads.
- `ui` page: static HTML/JS served by backend at `/`, displays metrics/overlays, and sends human instructions.

HITL control loop details:
- Feedback can include `paused=true/false` for explicit pause/resume.
- Forecast worker emits `forecast_topk` events with top-10 likely failure candidates.
- UI selects candidates from forecast queue (not random mistakes), draws ROI, and resumes training after intervention.
- XAI events include a `modality_focus` map to support future multimodal attribution views.

Current trainer/worker entrypoints:
- Torch: `coflect-hitl-trainer-torch`, `coflect-hitl-xai-worker-torch`
- TensorFlow/Keras: `coflect-hitl-trainer-tf`, `coflect-hitl-xai-worker-tf`
- Forecast (CPU): `coflect-hitl-forecast-worker --backend <torch|tensorflow>`

## Extending with New Framework Support

1. Add backend implementation in `coflect/backends/`.
2. Register it via `register_backend("name", Factory)`.
3. Add integration tests for trainer + worker paths.
4. Add version window to `SUPPORT_MATRIX.md`.

## Extending with New Modules

1. Add module package under `coflect/modules/<module_name>/`.
2. Add `ModuleSpec` registration in module `__init__.py`.
3. Provide module-specific backend/trainer/worker entrypoints.
4. Add CLI aliases if needed.

This layout keeps module logic isolated and backends reusable across modules.

## Example Folder Convention

Use `examples/<module>/` with:

1. Numbered notebooks for onboarding (`01_*.ipynb`).
2. Script counterparts for automation/CI.
3. A local README for module-specific prerequisites.

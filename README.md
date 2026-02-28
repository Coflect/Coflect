# Coflect
### Agnetic Deep Learning Framework
## HITL
### Human In Loop Training
Interactive Trainer Dashboard

A minimal, **non-blocking** human-in-the-loop training visualiser prototype.

**Key performance rule:** training stays fast because the trainer only emits lightweight JSON events and **requests XAI asynchronously**. Heavy explainability runs in a separate worker process.
Current release line is **Torch-first** with **TensorFlow/Keras support for HITL MVP paths**.
JAX remains scaffolded for staged rollout.

## Repo layout
- `coflect/backends/` backend adapter interfaces and implementations
- `coflect/modules/` pluggable module namespace (`hitl` module included)
- `coflect/modules/hitl/backend/` FastAPI server (events + WebSocket + feedback + XAI queue)
- `coflect/modules/hitl/trainer/` backend-specific training loops (Torch + TensorFlow/Keras)
- `coflect/modules/hitl/xai_worker/` async attribution workers
- `coflect/modules/hitl/backend/static/` zero-build browser UI served by FastAPI
- `cpp/kernels/` native kernel acceleration module (scaffold)
- `rust/encoder/` high-performance encoder module (scaffold)
- `examples/` reference examples and notebook walkthroughs

## 1) Setup (editable install)
Create venv and install deps:

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate
pip install -e .[server,dev]
```

Optional framework extras:

```bash
pip install -e .[tensorflow]
pip install -e .[jax]
```

## 2) Start server
```bash
coflect-hitl-backend --host 0.0.0.0 --port 8000
# or: python -m coflect.modules.hitl.backend --host 0.0.0.0 --port 8000
```

## 2a) One-command launch (recommended)
```bash
coflect-hitl-run --backend torch --dataset cifar10_catsdogs --data_root ./data --steps 5000 --xai_every 100 --forecast_every 20
# module fallback:
# python -m coflect.modules.hitl.launcher --backend torch --dataset cifar10_catsdogs --data_root ./data --steps 5000 --xai_every 100 --forecast_every 20
```
This starts backend + trainer + forecast worker + XAI worker together and writes logs under `./.coflect_logs/hitl/`.

## 3) Start trainer (new terminal)
```bash
coflect-hitl-trainer-torch --server http://localhost:8000 --steps 5000 --xai_every 250 --forecast_every 20
# or: python -m coflect.modules.hitl.trainer --server http://localhost:8000 --steps 5000 --xai_every 250
# optional: --mistake_every 40 (legacy explicit mistake-triggered overlays)
```

Torch real-data variant (CIFAR-10 cat vs dog):
```bash
coflect-hitl-trainer-torch --server http://localhost:8000 --dataset cifar10_catsdogs --data_root ./data --download_data --steps 5000 --xai_every 250 --forecast_every 20
```

TensorFlow/Keras variant:
```bash
coflect-hitl-trainer-tf --server http://localhost:8000 --steps 5000 --xai_every 250 --forecast_every 20
# or: python -m coflect.modules.hitl.trainer --backend tensorflow --server http://localhost:8000 --steps 5000 --xai_every 250
# optional: --mistake_every 40 (legacy explicit mistake-triggered overlays)
```

## 4) Start forecast worker (new terminal, CPU-only)
```bash
coflect-hitl-forecast-worker --server http://localhost:8000 --backend torch
# or: python -m coflect.modules.hitl.forecast --server http://localhost:8000 --backend torch
```

TensorFlow/Keras variant:
```bash
coflect-hitl-forecast-worker --server http://localhost:8000 --backend tensorflow
```

## 5) Start XAI worker (new terminal)
```bash
coflect-hitl-xai-worker-torch --server http://localhost:8000 --xai_method consensus
# or: python -m coflect.modules.hitl.xai_worker --server http://localhost:8000 --xai_method consensus
# single-GPU budget mode: add `--device cpu` to avoid trainer GPU contention
```

Torch real-data variant (must match trainer dataset config):
```bash
coflect-hitl-xai-worker-torch --server http://localhost:8000 --xai_method consensus --dataset cifar10_catsdogs --data_root ./data --download_data
```

TensorFlow/Keras variant:
```bash
coflect-hitl-xai-worker-tf --server http://localhost:8000 --xai_method consensus
# or: python -m coflect.modules.hitl.xai_worker --backend tensorflow --server http://localhost:8000 --xai_method consensus
```

## 6) Open UI
Open: http://localhost:8000

## Notes
- Default demo mode uses a deterministic synthetic dataset.
- Torch also supports a real dataset mode: `cifar10_catsdogs` (binary cat/dog subset of CIFAR-10).
- Trainer and XAI worker must use the same dataset config so `sample_idx` regeneration stays consistent.
- XAI worker supports `livecam` (alias: `gradcam`), `smoothgrad`, and `consensus` (LiveCAM + SmoothGrad blend). Default is `consensus` to reduce single-method artifacts.
- XAI payload includes top predicted classes/probabilities and an `xai_agreement` score between LiveCAM and SmoothGrad to reduce single-method misread risk.
- XAI view shows the overlay image with a thin yellow `focus_bbox` around the strongest activation region.
- XAI payload also includes a `modality_focus` map (currently `{"image": 1.0}`) so multimodal attribution splits can be introduced without protocol changes.
- Forecast worker runs on CPU, consumes compact telemetry, and opens review windows only when warmup/competence/plateau/stability gates pass.
- UI shows top-10 likely future failures (forecast candidates), supports pause/resume, sample selection, ROI drawing, and feedback submission.
- UI supports ROI box drawing on top of latest overlay; feedback sends normalized ROI coordinates to trainer.
- Feedback supports flexible free text parsing (`increase focus by 10%`, `focus center`, coordinate ROI commands), with optional explicit `strength` override.

## Examples
- Example index: `examples/README.md`
- HITL quickstart notebook: `examples/hitl/01_hitl_module_quickstart.ipynb`
- HITL script counterpart: `examples/hitl/run_hitl_demo.py`
- Long-run overhead benchmark script: `scripts/benchmark_hitl_overhead.py`

## Performance Benchmark (Long Run)
Run reproducible long-run overhead benchmarking:

```bash
PYTHONPATH=. python scripts/benchmark_hitl_overhead.py \
  --steps 200 \
  --repeats 3 \
  --warmup_runs 1 \
  --dataset synthetic \
  --device cpu \
  --num_workers 0 \
  --batch_size 64 \
  --output docs/benchmarks/hitl_overhead_longrun_2026-03-01.json
```

Latest recorded run (**March 1, 2026**):
- Artifact: `docs/benchmarks/hitl_overhead_longrun_2026-03-01.json`
- Environment: macOS 14.4.1 arm64, Python 3.11.14, Torch 2.8.0, CPU
- Config: synthetic dataset, 200 steps, 3 measured repeats, 1 warmup run
- Baseline mean: `105.81s` (`1.893 steps/s`)
- Coflect minimal mean: `105.41s` (`1.899 steps/s`)
- Mean slowdown (elapsed): `-0.37%` (no measurable slowdown; within run variance)

## Coding Standards
- Thin adapters and framework-agnostic core paths.
- Fail-soft trainer/worker networking to avoid training interruption.
- Explicit type hints and module/function docstrings for public modules.
- Keep heavy explainability compute outside trainer hot path.

## Packaging and Release
- Package metadata and tool configs: `pyproject.toml`
- Backend/framework support policy: `SUPPORT_MATRIX.md`
- Release notes and version history: GitHub Releases (`https://github.com/Coflect/Coflect/releases`)
- Contribution guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- CI workflow: `.github/workflows/ci.yml`
- TestPyPI workflow (GitHub pre-release/manual): `.github/workflows/publish-testpypi.yml`
- PyPI publish workflow (GitHub Release-triggered): `.github/workflows/publish-pypi.yml`
- Launch checklist: `docs/LAUNCH_CHECKLIST.md`
- Release playbook: `docs/RELEASE_PYPI.md`
- Architecture and extension guide: `docs/ARCHITECTURE.md`

## Maintainer Commands
```bash
make install-dev
make quality
make test
make build
make release-check
make smoke
# TensorFlow smoke path:
./scripts/smoke_hitl.sh 20 tensorflow
```

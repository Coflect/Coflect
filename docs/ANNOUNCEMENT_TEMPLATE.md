# Announcement Template

## Short Post (X/LinkedIn)

Coflect is now open source.

Coflect is a human-in-the-loop training framework built for speed-first workflows:
- Torch-first release
- CPU-only forecast worker for top-k likely future failures
- Async XAI worker (non-blocking trainer)
- Zero-build browser UI served directly by FastAPI (no npm runtime dependency)
- ROI-guided feedback loop
- Modular backend + module architecture for TensorFlow/Keras and JAX rollout

Repo: https://github.com/coflect/coflect
Docs: https://github.com/coflect/coflect/tree/main/docs

## Longer Post (Blog/README release note)

Today we are releasing Coflect v0.1.0.

What’s in this release:
- HITL module with backend, trainer, forecast worker, and async XAI worker
- Torch backend support and pluggable backend/module registries
- LiveCAM-based attribution stream and top-10 forecast review queue in UI
- PyPI packaging and CI/release workflows
- Example notebook + script quickstart

Roadmap highlights:
- TensorFlow/Keras backend adapter
- JAX backend adapter
- Native acceleration integration from C++ and Rust modules

If you are interested in practical human-in-the-loop training systems, we would love feedback and contributions.

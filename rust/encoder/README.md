# Rust Encoder Module

This module is reserved for fast, safe encoding/serialization paths (e.g., image/event payloads).

## Scope

- Fast PNG/JPEG/event encoding where Python overhead becomes measurable.
- Deterministic behavior and memory safety for worker-side pipelines.

## Design Principles

- Keep Python integration optional.
- Prefer small, focused crates and explicit perf benchmarks.
- Add FFI boundary tests before enabling by default.

## Initial Plan

1. Build baseline encoder crate and benchmark.
2. Add Python binding entrypoint (pyo3/maturin or cffi bridge).
3. Integrate behind feature flag in XAI worker.

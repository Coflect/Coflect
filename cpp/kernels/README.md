# C++/CUDA Kernels Module

This module is reserved for high-performance kernels used by Coflect hot paths.

## Scope

- Latency-sensitive tensor/image operations that benefit from native implementation.
- Optional integration into trainer or XAI worker through Python bindings.

## Design Principles

- Keep trainer loop non-blocking.
- Prefer isolated native calls with explicit benchmarks.
- Ship feature flags so native acceleration can be enabled/disabled safely.

## Initial Plan

1. Add benchmark harness against Python/Torch baseline.
2. Implement first kernel with correctness tests.
3. Add build pipeline for Linux/macOS.

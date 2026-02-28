# Support Matrix

This matrix defines what Coflect commits to test and support per release line.

## Runtime Targets

- Python: 3.10, 3.11, 3.12
- OS: Linux (primary), macOS (best effort), Windows (best effort)

## Backend Support Policy

- Torch is the primary backend for `0.x` releases.
- TensorFlow/Keras is available for HITL MVP trainer + XAI worker paths in `0.1.x` as experimental support.
- JAX remains staged behind adapter interfaces until integration tests are in place.
- We follow an `N-2` style policy where practical: support the latest stable backend version plus previous compatible versions.

## Initial Compatibility Window (`v0.1.x`)

As of **February 27, 2026**, this is the target compatibility range we release and test against.

- PyTorch: `>=2.4,<2.9`
- torchvision: `>=0.19,<0.24`
- TensorFlow (experimental): `>=2.17,<2.21`
- Keras (experimental): `>=3.0,<4`
- JAX/JAXLIB (planned): `>=0.4.30,<0.8`

Upstream release streams:
- PyTorch: https://pypi.org/project/torch/
- torchvision: https://pypi.org/project/torchvision/
- TensorFlow: https://pypi.org/project/tensorflow/
- JAX: https://pypi.org/project/jax/
- jaxlib: https://pypi.org/project/jaxlib/

## Upgrade Rules

- Minor-version support updates are done in minor releases.
- Breaking backend support changes require a minor release note and migration guidance.
- Any trainer hot-path perf regression above ~3% needs explicit justification and opt-in controls.

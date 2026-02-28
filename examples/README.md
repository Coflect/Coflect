# Examples

This folder follows a structure commonly used in large ML repositories (PyTorch examples, TensorFlow/Keras examples, JAX examples):

- Keep examples grouped by product/module (`examples/<module>/`).
- Keep onboarding artifacts first (`01_*.ipynb`).
- Pair notebooks with script equivalents for CI-friendliness.
- Keep examples deterministic, lightweight, and explicit about prerequisites.

## Layout

- `hitl/`
  - `README.md` module-specific quickstart
  - `01_hitl_module_quickstart.ipynb` notebook walkthrough
  - `run_hitl_demo.py` script counterpart

## Authoring Rules

- Do not hide required setup in notebook state.
- Prefer CLI commands that map to package entrypoints.
- Include clean shutdown guidance when examples launch local processes.

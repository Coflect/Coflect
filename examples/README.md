# Examples

This folder follows a structure commonly used in large ML repositories (PyTorch examples, TensorFlow/Keras examples, JAX examples):

- Keep examples grouped by product/module (`examples/<module>/`).
- Keep onboarding artifacts first (`01_*.ipynb`).
- Pair notebooks with script equivalents for CI-friendliness.
- Keep examples deterministic, lightweight, and explicit about prerequisites.

## Layout

- `hilt/`
  - `README.md` module-specific quickstart
  - `01_hilt_module_quickstart.ipynb` Torch workflow notebook
  - `02_hilt_tensorflow_keras_workflow.ipynb` TensorFlow/Keras workflow notebook
  - `run_hilt_demo.py` script counterpart

## Authoring Rules

- Do not hide required setup in notebook state.
- Prefer CLI commands that map to package entrypoints.
- Include clean shutdown guidance when examples launch local processes.

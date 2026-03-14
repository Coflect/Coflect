# HILT Examples

## Files

- `01_hilt_module_quickstart.ipynb`: Torch day-to-day experiment workflow notebook.
- `02_hilt_tensorflow_keras_workflow.ipynb`: TensorFlow/Keras day-to-day experiment workflow notebook.
- `coflect_tf_xai_worker_local.py`: TensorFlow local XAI worker helper used by notebook bridge.
- `run_hilt_demo.py`: script helper that prints (or launches) backend/trainer/forecast/XAI commands.

## Setup

```bash
pip install coflect
# optional framework extras:
# pip install "coflect[tensorflow]"
```

Editable install for local development:

```bash
pip install -e .
# optional contributor tooling:
# pip install -e .[dev]
```

For TensorFlow/Keras runtime commands:

```bash
pip install -e .[tensorflow]
```

## Torch Path

Notebook:
- Open `01_hilt_module_quickstart.ipynb` and run cells top-to-bottom.
- Notebook bridge auto-starts backend + XAI worker for live UI updates.
- Continue in the UI at `http://127.0.0.1:8000`.

Script:

```bash
python examples/hilt/run_hilt_demo.py \
  --backend torch \
  --dataset cifar10_catsdogs \
  --data-root ./data \
  --download-data
```

One-command launcher equivalent:

```bash
coflect-hilt-run --backend torch --dataset cifar10_catsdogs --data_root ./data --download_data --steps 1000 --xai_every 100 --forecast_every 20
```

## TensorFlow/Keras Path

Notebook:
- Open `02_hilt_tensorflow_keras_workflow.ipynb` and run cells top-to-bottom.
- Notebook bridge can use `coflect_tf_xai_worker_local.py` when present for dataset-aligned live overlays.
- Continue in the UI at `http://127.0.0.1:8000`.

Script:

```bash
python examples/hilt/run_hilt_demo.py --backend tensorflow
```

One-command launcher equivalent:

```bash
coflect-hilt-run --backend tensorflow --steps 1000 --xai_every 100 --forecast_every 20
```

## Open UI

Open `http://127.0.0.1:8000`.

Use the review window to pick a forecasted sample, draw ROI, submit feedback, then resume training.

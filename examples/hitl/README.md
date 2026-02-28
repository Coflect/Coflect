# HITL Examples

## Files

- `01_hitl_module_quickstart.ipynb`: notebook walkthrough.
- `run_hitl_demo.py`: script that prints/optionally launches local processes.

## Quick Start

1. Install package and server deps:

```bash
pip install -e .[server,dev]
```

For TensorFlow/Keras runtime commands, also install:

```bash
pip install -e .[tensorflow]
```

2. Start processes (or use the script in this folder):

```bash
coflect-hitl-run --backend torch --dataset cifar10_catsdogs --data_root ./data --steps 1000 --xai_every 100 --forecast_every 20
# module fallback:
# python -m coflect.modules.hitl.launcher --backend torch --dataset cifar10_catsdogs --data_root ./data --steps 1000 --xai_every 100 --forecast_every 20
```

Equivalent manual split:

```bash
coflect-hitl-backend --host 0.0.0.0 --port 8000
coflect-hitl-trainer-torch --server http://localhost:8000 --steps 1000 --xai_every 100 --forecast_every 20
coflect-hitl-forecast-worker --server http://localhost:8000 --backend torch
coflect-hitl-xai-worker-torch --server http://localhost:8000 --xai_method consensus
```

Torch real-data variant (CIFAR-10 cat vs dog):

```bash
coflect-hitl-trainer-torch --server http://localhost:8000 --dataset cifar10_catsdogs --data_root ./data --download_data --steps 1000 --xai_every 100 --forecast_every 20
coflect-hitl-forecast-worker --server http://localhost:8000 --backend torch
coflect-hitl-xai-worker-torch --server http://localhost:8000 --xai_method consensus --dataset cifar10_catsdogs --data_root ./data --download_data
```

TensorFlow/Keras variant:

```bash
coflect-hitl-trainer-tf --server http://localhost:8000 --steps 1000 --xai_every 100 --forecast_every 20
coflect-hitl-forecast-worker --server http://localhost:8000 --backend tensorflow
coflect-hitl-xai-worker-tf --server http://localhost:8000 --xai_method consensus
```

3. Open UI:

Open `http://localhost:8000`.

In UI, wait for the review window to open, select one of the top forecasted failure samples, draw ROI, send feedback, then resume.

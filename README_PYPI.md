# Coflect

Coflect is a Human In Loop Training (HILT) deep learning framework for live training control with non-blocking explainability.

Torch is the primary production path. TensorFlow/Keras and JAX are available as optional extras.

## Install

```bash
pip install coflect
```

Optional extras:

```bash
pip install "coflect[tensorflow]"
pip install "coflect[jax]"
```

Python requirement: `>=3.10`.

## Quickstart

```bash
coflect-hilt-run \
  --backend torch \
  --dataset cifar10_catsdogs \
  --data_root ./data \
  --download_data \
  --steps 1000 \
  --xai_every 100 \
  --forecast_every 20
```

Open [http://localhost:8000](http://localhost:8000).

UI-only server command:

```bash
coflect-hilt-ui --host 127.0.0.1 --port 8000
```

## Project links

- Source code: [github.com/Coflect/Coflect](https://github.com/Coflect/Coflect)
- Documentation and examples: [github.com/Coflect/Coflect/tree/main/examples](https://github.com/Coflect/Coflect/tree/main/examples)
- Issue tracker: [github.com/Coflect/Coflect/issues](https://github.com/Coflect/Coflect/issues)
- Releases: [github.com/Coflect/Coflect/releases](https://github.com/Coflect/Coflect/releases)

License: Apache-2.0

.PHONY: install-dev quality test build release-check smoke benchmark-hitl

install-dev:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev,server]

quality:
	ruff check .
	mypy coflect

test:
	pytest

build:
	python -m build

release-check:
	./scripts/release_check.sh

smoke:
	./scripts/smoke_hitl.sh 20

benchmark-hitl:
	PYTHONPATH=. python scripts/benchmark_hitl_overhead.py --steps 200 --repeats 3 --warmup_runs 1 --dataset synthetic --device cpu --num_workers 0 --batch_size 64 --output docs/benchmarks/hitl_overhead_longrun_latest.json

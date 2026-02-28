#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run() {
  echo "\n==> $*"
  "$@"
}

run python -m pip --version
run python -m pip install --upgrade pip
run python -m pip install -e .[dev,server]
run ruff check .
run mypy coflect
run pytest
run python -m build

echo "\nRelease checks passed."

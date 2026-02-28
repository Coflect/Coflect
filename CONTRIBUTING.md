# Contributing to Coflect

Thanks for contributing.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,server]
```

## Quality Bar

Before opening a PR, run:

```bash
ruff check .
mypy coflect
pytest
python -m build
```

Or run the bundled checks:

```bash
make release-check
```

## Coding Guidelines

- Keep trainer hot path minimal and non-blocking.
- Keep heavy XAI in worker processes.
- Prefer explicit type hints and short, precise docstrings.
- Add tests for behavior changes (especially deterministic behavior and protocol changes).
- Avoid introducing heavyweight dependencies in core paths.

## Pull Request Checklist

- Include motivation and behavior changes.
- Include performance impact (SPS delta) when touching training/XAI hot paths.
- Update docs for any protocol, CLI, or config change.
- Keep compatibility notes in `SUPPORT_MATRIX.md` when backend support changes.

## Release Notes

For release-impacting changes, add a short note in the PR body:

- Added/changed CLI commands
- Added/changed websocket events
- Added/changed backend support

## Community Standards

- Code of Conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`

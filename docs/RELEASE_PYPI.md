# PyPI Release Process

This process follows common patterns used by large OSS repos (GitHub Releases, CI quality gates, reproducible builds, and automated publish).
GitHub Releases are the canonical release notes/history for this repository.

## 1) Prepare release

- Update `pyproject.toml` version.
- Update `SUPPORT_MATRIX.md` if backend/version windows changed.
- Ensure CI is green on main.
- Prepare GitHub release notes draft (features, fixes, perf impact, compatibility notes).
- Run launch readiness checklist: `docs/LAUNCH_CHECKLIST.md`

## 2) Create GitHub release

- Create annotated tag: `vX.Y.Z`.
- Publish GitHub Release with notes:
  - features
  - fixes
  - perf impact (if trainer/XAI paths changed)
  - compatibility changes

Recommended cadence (similar to major ML OSS repos):
- `main` for ongoing development
- `release/x.y` branch only for patch fixes on that minor line
- Optional pre-releases (`vX.Y.Zrc1`) for dependency or API-risky updates

## 3) Automated publish

### 3a) Pre-release to TestPyPI

- Workflow: `.github/workflows/publish-testpypi.yml`
- Trigger: GitHub Release `published` where `pre-release=true`, or manual dispatch.
- Build: `python -m build`
- Publish target: TestPyPI
- Validate install:

```bash
pip install --index-url https://test.pypi.org/simple/ coflect==X.Y.Zrc1
```

### 3b) Stable release to PyPI

- Workflow: `.github/workflows/publish-pypi.yml`
- Trigger: GitHub Release `published` where `pre-release=false`.
- Build: `python -m build`
- Publish: `pypa/gh-action-pypi-publish` via trusted publisher (OIDC).

## 4) Post-release checks

- Verify package on PyPI (`pip install coflect==X.Y.Z`).
- Smoke test entrypoints:
  - `coflect-hitl-backend`
  - `coflect-hitl-trainer-torch`
  - `coflect-hitl-forecast-worker`
  - `coflect-hitl-xai-worker-torch`
- Open next development milestone and set next version.

## Versioning

- Use semantic versioning:
  - `MAJOR`: incompatible public API/protocol changes
  - `MINOR`: backward-compatible features
  - `PATCH`: bug fixes only
- For `0.x`, treat minor bumps as potentially breaking and call out migration notes explicitly.

## Trusted Publisher Setup

In PyPI project settings:
- Add GitHub trusted publisher for this repository.
- Match workflow file and environment constraints used for release.

Avoid long-lived API tokens when possible.

# Launch Checklist

## Product Readiness

- [ ] End-to-end HITL flow validated (backend, trainer, forecast worker, XAI worker, UI)
- [ ] ROI feedback path verified
- [ ] Snapshot sync path verified
- [ ] XAI budget mechanism validated
- [ ] Training throughput regression measured and documented

## Engineering Quality

- [ ] `make release-check` passes
- [ ] `make smoke` passes
- [ ] CI green on main
- [ ] Changelog updated for release

## Release Operations

- [ ] PyPI trusted publisher configured
- [ ] TestPyPI publish validated (`vX.Y.Zrc1`)
- [ ] PyPI release tag created (`vX.Y.Z`)
- [ ] GitHub release notes published

## Announcement

- [ ] Demo GIF/video captured
- [ ] Website/repo links verified
- [ ] Announcement copy reviewed and posted

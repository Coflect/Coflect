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
- [ ] GitHub release notes draft prepared

## Release Operations

- [ ] PyPI trusted publisher configured
- [ ] TestPyPI publish validated from GitHub pre-release
- [ ] GitHub release published with tag (`vX.Y.Z`)

## Announcement

- [ ] Demo GIF/video captured
- [ ] Website/repo links verified
- [ ] Announcement copy reviewed and posted

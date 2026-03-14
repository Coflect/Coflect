"""Numpy synthetic data helpers shared by non-PyTorch runtimes.

This dataset is intentionally synthetic (non-medical) and deterministic by `idx`.
Samples contain clear geometric structure so attribution overlays are interpretable.
"""

from __future__ import annotations

import numpy as np


def _class_color(class_idx: int) -> np.ndarray:
    """Return a deterministic RGB color in [0, 1] for a class index."""
    phase = float(class_idx) * 0.91
    angles = np.array([0.0, 2.1, 4.2], dtype=np.float32)
    base = 0.5 + 0.5 * np.sin(phase + angles)
    return (0.35 + 0.65 * base).astype(np.float32)


def _gaussian_blob(height: int, width: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Return a 2D gaussian blob centered at (cx, cy)."""
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)


def _shape_mask(height: int, width: int, cx: float, cy: float, size: float, shape_id: int) -> np.ndarray:
    """Return a soft binary shape mask in [0, 1]."""
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    dx = xs - cx
    dy = ys - cy
    adx = np.abs(dx)
    ady = np.abs(dy)

    sid = int(shape_id % 4)
    if sid == 0:
        core = ((dx * dx + dy * dy) <= (size * size)).astype(np.float32)  # disk
    elif sid == 1:
        core = np.logical_and(adx <= size, ady <= size).astype(np.float32)  # square
    elif sid == 2:
        core = ((adx + ady) <= (size * 1.15)).astype(np.float32)  # diamond
    else:
        arm = max(1.2, size * 0.34)
        core = np.logical_or(
            np.logical_and(adx <= arm, ady <= size),
            np.logical_and(ady <= arm, adx <= size),
        ).astype(np.float32)  # cross

    soft = _gaussian_blob(height, width, cx=cx, cy=cy, sigma=max(1.2, size * 0.35))
    return np.clip(core * 0.65 + soft * 0.35, 0.0, 1.0).astype(np.float32)


def sample_chw_by_idx(idx: int, num_classes: int = 10, image_size: int = 64) -> tuple[np.ndarray, int]:
    """Return deterministic CHW image and label for a sample index."""
    idx = int(idx)
    rng = np.random.default_rng(seed=idx)
    label = int(idx % max(1, num_classes))

    h = int(image_size)
    w = int(image_size)
    img = np.zeros((3, h, w), dtype=np.float32)

    # Smooth non-medical background with a gentle gradient.
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    bg = 0.08 + 0.10 * xs + 0.06 * ys
    img += np.stack([bg * 0.9, bg, bg * 1.1], axis=0)

    # Class-conditioned primary shape placed on a ring around image center.
    theta = (2.0 * np.pi * float(label)) / float(max(1, num_classes))
    radius = 0.25 * float(min(h, w))
    cx = (0.5 * float(w)) + radius * float(np.cos(theta)) + float(rng.normal(0.0, 1.3))
    cy = (0.5 * float(h)) + radius * float(np.sin(theta)) + float(rng.normal(0.0, 1.3))
    size = max(4.0, 0.11 * float(min(h, w)))
    main_mask = _shape_mask(h, w, cx=cx, cy=cy, size=size, shape_id=label)
    main_color = _class_color(label)[:, None, None]
    img += (1.35 * main_mask[None, :, :]) * main_color

    # Weak distractor to avoid over-trivial training while keeping readability.
    d_cx = float(rng.uniform(0.15 * w, 0.85 * w))
    d_cy = float(rng.uniform(0.15 * h, 0.85 * h))
    distractor = _gaussian_blob(h, w, cx=d_cx, cy=d_cy, sigma=max(1.8, size * 0.45))
    distractor_color = _class_color((label + 2) % max(1, num_classes))[:, None, None]
    img += (0.25 * distractor[None, :, :]) * distractor_color

    # Light pixel noise keeps signal realistic but still human-readable.
    img += rng.normal(0.0, 0.02, size=(3, h, w)).astype(np.float32)

    img = np.clip(img, -1.0, 1.0).astype(np.float32, copy=False)
    return img, label


def chw_to_hwc(img_chw: np.ndarray) -> np.ndarray:
    """Convert CHW float image to HWC float image."""
    return np.transpose(img_chw, (1, 2, 0)).astype(np.float32, copy=False)

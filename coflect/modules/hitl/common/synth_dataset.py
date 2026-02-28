"""Deterministic synthetic dataset used by trainer and XAI worker."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from coflect.modules.hitl.common.synth_numpy import sample_chw_by_idx


class DeterministicSyntheticImages(Dataset):
    """Synthetic samples where index `i` always yields identical image + label."""

    def __init__(self, n: int = 50_000, num_classes: int = 10, image_size: int = 64):
        """Create a deterministic synthetic dataset view.

        Example:
            >>> ds = DeterministicSyntheticImages(n=100, num_classes=10, image_size=64)
        """
        self.n = n
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        """Return dataset cardinality.

        Example:
            >>> len(DeterministicSyntheticImages(n=7))
            7
        """
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return `(image_chw, label, sample_idx)` for a deterministic index.

        Example:
            >>> img, y, sid = DeterministicSyntheticImages(n=8)[3]
        """
        img, label = sample_chw_by_idx(
            idx=idx,
            num_classes=self.num_classes,
            image_size=self.image_size,
        )
        return torch.from_numpy(img), torch.tensor(int(label), dtype=torch.long), int(idx)

"""Torch dataset builders for HITL training and XAI regeneration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from coflect.modules.hitl.common.synth_dataset import DeterministicSyntheticImages

DatasetName = Literal["synthetic", "cifar10_catsdogs"]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for building deterministic dataset instances."""

    name: DatasetName
    root: str
    split: Literal["train", "test"] = "train"
    download_data: bool = False


class Cifar10CatsDogs(Dataset):
    """Binary subset of CIFAR-10 containing only cat vs dog classes.

    Labels are remapped to:
    - 0: cat (CIFAR class 3)
    - 1: dog (CIFAR class 5)

    Returned `sample_idx` is the local subset index so trainer and XAI worker
    can regenerate the same sample by index deterministically.
    """

    CAT = 3
    DOG = 5

    def __init__(self, root: str, split: Literal["train", "test"] = "train", download_data: bool = False) -> None:
        """Load CIFAR-10 and filter into cat-vs-dog subset.

        Example:
            >>> ds = Cifar10CatsDogs(root=\"./data\", split=\"train\", download_data=False)
        """
        train = split == "train"
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.base = CIFAR10(root=root, train=train, download=download_data, transform=tfm)
        targets = [int(t) for t in self.base.targets]
        self.indices = [i for i, t in enumerate(targets) if t in (self.CAT, self.DOG)]
        if not self.indices:
            raise RuntimeError("CIFAR-10 cat/dog subset is empty; verify dataset files.")

        self._targets = targets
        self.num_classes = 2
        self.image_size = 32

    def __len__(self) -> int:
        """Return number of subset samples.

        Example:
            >>> n = len(ds)
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return `(image_chw, binary_label, local_subset_idx)`.

        Example:
            >>> img, y, sid = ds[0]
        """
        local_idx = int(idx)
        base_idx = self.indices[local_idx]
        img, raw_label = self.base[base_idx]
        raw = int(raw_label)
        label = 0 if raw == self.CAT else 1
        return img, torch.tensor(label, dtype=torch.long), local_idx


def build_torch_dataset(cfg: DatasetConfig) -> Dataset:
    """Construct a HITL dataset compatible with trainer and XAI worker."""
    if cfg.name == "synthetic":
        return DeterministicSyntheticImages(n=100_000, num_classes=10, image_size=64)
    if cfg.name == "cifar10_catsdogs":
        return Cifar10CatsDogs(root=cfg.root, split=cfg.split, download_data=cfg.download_data)
    raise ValueError(f"Unsupported dataset: {cfg.name}")

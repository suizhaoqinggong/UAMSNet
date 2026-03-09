"""WORD dataset adapter for medical image segmentation."""
from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from framework.contracts import Batch, DataAdapter, DatasetLike, Sample


def mask_to_onehot(mask: np.ndarray, palette: list) -> np.ndarray:
    """Convert segmentation mask to one-hot encoding.

    Args:
        mask: Mask array of shape (H, W, C) where C is usually 1 or 3
        palette: List of class indices

    Returns:
        One-hot encoded mask of shape (H, W, K) where K is num_classes
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """Row in WORD dataset manifest."""

    sample_id: str
    image_path: str
    mask_path: str
    split: str


class WORDDataset(Dataset[Sample]):
    """WORD dataset for medical image segmentation."""

    def __init__(
        self,
        *,
        rows: Sequence[ManifestRow],
        split: str,
        num_classes: int = 17,
        palette: Sequence[int] | None = None,
    ) -> None:
        self._rows = list(rows)
        self._split = split
        self._num_classes = num_classes
        self._palette = list(palette) if palette is not None else list(range(num_classes))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Sample:
        row = self._rows[index]

        # Load image and mask from .npy files
        image = np.load(row.image_path)  # Shape: (H, W, C)
        mask = np.load(row.mask_path)  # Shape: (H, W)

        # Transpose image to (C, H, W)
        image = image.transpose((2, 0, 1))

        # Convert mask to one-hot encoding
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        mask = mask_to_onehot(mask, self._palette)
        mask = mask.transpose((2, 0, 1))  # (num_classes, H, W)

        # Convert to float32 tensors
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))

        return {
            "image": image,
            "label": mask,
            "id": row.sample_id,
            "meta": {
                "split": self._split,
                "image_path": row.image_path,
                "mask_path": row.mask_path,
            },
        }


class WORDDataAdapter(DataAdapter):
    """Data adapter for WORD medical image segmentation dataset."""

    def __init__(
        self,
        *,
        processed_root: str = "data/processed/word",
        train_manifest: str = "train.csv",
        val_manifest: str = "val.csv",
        test_manifest: str = "test.csv",
        num_classes: int = 17,
        batch_size: int = 4,
    ) -> None:
        self._processed_root = Path(processed_root)
        self._train_manifest_name = train_manifest
        self._val_manifest_name = val_manifest
        self._test_manifest_name = test_manifest
        self._num_classes = num_classes
        self._batch_size = batch_size

        self._train_dataset: WORDDataset | None = None
        self._val_dataset: WORDDataset | None = None
        self._test_dataset: WORDDataset | None = None

    def prepare(self) -> None:
        """Prepare datasets by loading manifests."""
        train_rows = self._load_manifest(self._train_manifest_name)
        val_rows = self._load_manifest(self._val_manifest_name)
        test_rows = self._load_manifest(self._test_manifest_name)

        self._train_dataset = WORDDataset(
            rows=train_rows,
            split="train",
            num_classes=self._num_classes,
        )
        self._val_dataset = WORDDataset(
            rows=val_rows,
            split="val",
            num_classes=self._num_classes,
        )
        self._test_dataset = WORDDataset(
            rows=test_rows,
            split="test",
            num_classes=self._num_classes,
        )

    def _load_manifest(self, manifest_name: str) -> list[ManifestRow]:
        """Load manifest CSV file."""
        manifest_path = self._processed_root / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        rows = []
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    ManifestRow(
                        sample_id=row["sample_id"],
                        image_path=row["image_path"],
                        mask_path=row["mask_path"],
                        split=row["split"],
                    )
                )
        return rows

    def get_splits(self) -> tuple[DatasetLike, DatasetLike, DatasetLike]:
        """Get train, validation, and test datasets."""
        if self._train_dataset is None:
            raise RuntimeError("prepare() must be called before get_splits()")
        return self._train_dataset, self._val_dataset, self._test_dataset

    def collate_fn(self, batch: list[Sample]) -> Batch:
        """Collate samples into a batch."""
        images = torch.stack([sample["image"] for sample in batch])
        labels = torch.stack([sample["label"] for sample in batch])
        ids = [sample["id"] for sample in batch]

        batch_dict: Batch = {
            "image": images,
            "label": labels,
            "id": ids,
        }

        # Add metadata if present
        if "meta" in batch[0]:
            batch_dict["meta"] = {
                key: [sample["meta"][key] for sample in batch]
                for key in batch[0]["meta"]
            }

        return batch_dict

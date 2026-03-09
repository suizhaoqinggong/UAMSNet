from __future__ import annotations

from typing import Protocol, TypeAlias

from torch.utils.data import Dataset

from .types import Batch, Sample

DatasetLike: TypeAlias = Dataset[Sample]


class DataAdapter(Protocol):
    def prepare(self) -> None: ...

    def get_splits(self) -> tuple[DatasetLike, DatasetLike, DatasetLike]: ...

    def collate_fn(self, batch: list[Sample]) -> Batch: ...

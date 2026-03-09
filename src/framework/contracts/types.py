from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from torch import Tensor

Metadata: TypeAlias = Mapping[str, Any]


class Sample(TypedDict):
    image: Tensor
    label: Tensor
    id: str
    meta: NotRequired[Metadata]


class Batch(TypedDict):
    image: Tensor
    label: Tensor
    id: list[str] | Tensor
    meta: NotRequired[Metadata]


ModelOutput: TypeAlias = Tensor
Predictions: TypeAlias = Tensor
Targets: TypeAlias = Tensor
LossValue: TypeAlias = Tensor
MetricResults: TypeAlias = dict[str, float]
ProblemType: TypeAlias = Literal["multilabel", "multiclass_segmentation"]

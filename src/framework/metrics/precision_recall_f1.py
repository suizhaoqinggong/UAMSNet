from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias, Literal

import torch
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

from framework.contracts import Metric, MetricResults

from ._helpers import prepare_metric_inputs, scalar_metric_value

AverageMethod: TypeAlias = Literal["micro", "macro"]


@dataclass(slots=True)
class PrecisionRecallF1Metric(Metric):
    num_classes: int
    average: AverageMethod = "macro"
    threshold: float = 0.5
    name: str = "precision_recall_f1"
    higher_is_better: bool = True
    _precision: TorchMetric = field(init=False)
    _recall: TorchMetric = field(init=False)
    _f1: TorchMetric = field(init=False)

    def __post_init__(self) -> None:
        if self.average not in {"micro", "macro"}:
            raise ValueError(f"Unsupported averaging mode: {self.average!r}")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be > 1 for multilabel precision/recall/f1.")
        self._precision = MultilabelPrecision(
            num_labels=self.num_classes,
            average=self.average,
            threshold=self.threshold,
        )
        self._recall = MultilabelRecall(
            num_labels=self.num_classes,
            average=self.average,
            threshold=self.threshold,
        )
        self._f1 = MultilabelF1Score(
            num_labels=self.num_classes,
            average=self.average,
            threshold=self.threshold,
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        metric_preds, metric_targets = prepare_metric_inputs(preds, targets)
        self._precision.update(metric_preds, metric_targets)
        self._recall.update(metric_preds, metric_targets)
        self._f1.update(metric_preds, metric_targets)

    def compute(self) -> MetricResults:
        return {
            "precision": scalar_metric_value(self._precision.compute()),
            "recall": scalar_metric_value(self._recall.compute()),
            "f1": scalar_metric_value(self._f1.compute()),
        }

    def reset(self) -> None:
        self._precision.reset()
        self._recall.reset()
        self._f1.reset()

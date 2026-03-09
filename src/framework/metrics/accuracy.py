from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import MultilabelAccuracy

from framework.contracts import Metric, MetricResults

from ._helpers import prepare_metric_inputs, scalar_metric_value


@dataclass(slots=True)
class AccuracyMetric(Metric):
    num_classes: int
    threshold: float = 0.5
    name: str = "accuracy"
    higher_is_better: bool = True
    _metric: TorchMetric = field(init=False)

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("num_classes must be > 1 for multilabel accuracy.")
        self._metric = MultilabelAccuracy(num_labels=self.num_classes, threshold=self.threshold)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        metric_preds, metric_targets = prepare_metric_inputs(preds, targets)
        self._metric.update(metric_preds, metric_targets)

    def compute(self) -> MetricResults:
        value = self._metric.compute()
        return {self.name: scalar_metric_value(value)}

    def reset(self) -> None:
        self._metric.reset()

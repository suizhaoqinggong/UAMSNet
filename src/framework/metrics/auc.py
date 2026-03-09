from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
)

from framework.contracts import Metric, MetricResults

from ._helpers import prepare_metric_inputs, scalar_metric_value


@dataclass(slots=True)
class AUCMetric(Metric):
    num_classes: int
    name: str = "auc"
    higher_is_better: bool = True
    _auroc: TorchMetric = field(init=False)
    _auprc: TorchMetric = field(init=False)

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("num_classes must be > 1 for multilabel AUC metrics.")
        self._auroc = MultilabelAUROC(num_labels=self.num_classes, average="macro")
        self._auprc = MultilabelAveragePrecision(num_labels=self.num_classes, average="macro")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        metric_preds, metric_targets = prepare_metric_inputs(preds, targets)
        self._auroc.update(metric_preds, metric_targets)
        self._auprc.update(metric_preds, metric_targets)

    def compute(self) -> MetricResults:
        auroc = scalar_metric_value(self._auroc.compute())
        auprc = scalar_metric_value(self._auprc.compute())
        return {
            "auroc": auroc,
            "auprc": auprc,
        }

    def reset(self) -> None:
        self._auroc.reset()
        self._auprc.reset()

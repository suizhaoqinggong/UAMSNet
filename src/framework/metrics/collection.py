from __future__ import annotations
from typing import TypeAlias

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from framework.contracts import Metric, MetricResults

MetricFactory: TypeAlias = Callable[[], Metric]


class MetricCollection:
    def __init__(self, metrics: Sequence[Metric], *, prefix: str | None = None) -> None:
        self._metrics = list(metrics)
        self._prefix = prefix.strip("/") if prefix is not None else None
        self.reset()

    @property
    def metrics(self) -> tuple[Metric, ...]:
        return tuple(self._metrics)

    def reset(self) -> None:
        for metric in self._metrics:
            metric.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        for metric in self._metrics:
            metric.update(preds, targets)

    def compute(self) -> MetricResults:
        computed: MetricResults = {}
        for metric in self._metrics:
            for key, value in metric.compute().items():
                output_key = self._format_key(key)
                if output_key in computed:
                    raise ValueError(f"Duplicate metric key detected: {output_key}")
                computed[output_key] = value
        return computed

    def _format_key(self, key: str) -> str:
        if self._prefix is None or self._prefix == "":
            return key
        return f"{self._prefix}/{key}"


@dataclass(frozen=True, slots=True)
class SplitMetricCollections:
    train: MetricCollection
    val: MetricCollection
    test: MetricCollection | None = None


def build_split_metric_collections(
    metric_factories: Sequence[MetricFactory],
    *,
    include_test: bool = True,
) -> SplitMetricCollections:
    train_metrics = MetricCollection([factory() for factory in metric_factories], prefix="train")
    val_metrics = MetricCollection([factory() for factory in metric_factories], prefix="val")
    test_metrics = (
        MetricCollection([factory() for factory in metric_factories], prefix="test")
        if include_test
        else None
    )
    return SplitMetricCollections(train=train_metrics, val=val_metrics, test=test_metrics)

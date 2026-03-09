from __future__ import annotations

from typing import Protocol

from .types import MetricResults, Predictions, Targets


class Metric(Protocol):
    name: str
    higher_is_better: bool

    def update(self, preds: Predictions, targets: Targets) -> None: ...

    def compute(self) -> MetricResults: ...

    def reset(self) -> None: ...

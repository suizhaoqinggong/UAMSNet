from __future__ import annotations

from typing import Protocol

from .types import MetricResults


class Runner(Protocol):
    def train(self) -> MetricResults: ...

    def validate(self, checkpoint_path: str | None = None) -> MetricResults: ...

    def test(self, checkpoint_path: str | None = None) -> MetricResults: ...

    def fit(self) -> MetricResults: ...

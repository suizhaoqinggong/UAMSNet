from __future__ import annotations

from typing import Literal, Protocol

from .types import Batch, LossValue, ModelOutput, Predictions, Targets

ProblemType = Literal["multilabel"]


class Task(Protocol):
    def compute_loss(self, outputs: ModelOutput, batch: Batch) -> LossValue: ...

    def extract_targets(self, batch: Batch) -> Targets: ...

    def postprocess_outputs(self, outputs: ModelOutput) -> Predictions: ...

    def infer_problem_type(self) -> ProblemType: ...

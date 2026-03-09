from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol

from torch.nn import Module
from torch.nn.parameter import Parameter

from .types import Batch, ModelOutput


class ModelAdapter(Protocol):
    def forward(self, batch: Batch) -> ModelOutput: ...

    def train(self, mode: bool = True) -> Module: ...

    def eval(self) -> Module: ...

    def to(self, device: object) -> Module: ...

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]: ...

    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        strict: bool = True,
        assign: bool = False,
    ) -> object: ...

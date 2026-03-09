from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch
from matplotlib.figure import Figure


class ExperimentLogger(Protocol):
    def start_run(self, *, run_name: str | None = None) -> None: ...

    def end_run(self, *, status: str = "FINISHED") -> None: ...

    def log_params(self, params: Mapping[str, object]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None: ...

    def log_artifact(self, artifact_path: str) -> None: ...

    def log_text(self, tag: str, text: str, *, step: int | None = None) -> None: ...

    def log_histogram(self, tag: str, values: torch.Tensor, *, step: int | None = None) -> None: ...

    def log_figure(
        self,
        tag: str,
        figure: Figure | list[Figure],
        *,
        step: int | None = None,
    ) -> None: ...


class NullExperimentLogger(ExperimentLogger):
    def start_run(self, *, run_name: str | None = None) -> None:
        del run_name

    def end_run(self, *, status: str = "FINISHED") -> None:
        del status

    def log_params(self, params: Mapping[str, object]) -> None:
        del params

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        del metrics
        del step

    def log_artifact(self, artifact_path: str) -> None:
        del artifact_path

    def log_text(self, tag: str, text: str, *, step: int | None = None) -> None:
        del tag
        del text
        del step

    def log_histogram(self, tag: str, values: torch.Tensor, *, step: int | None = None) -> None:
        del tag
        del values
        del step

    def log_figure(
        self,
        tag: str,
        figure: Figure | list[Figure],
        *,
        step: int | None = None,
    ) -> None:
        del tag
        del figure
        del step


def flatten_mapping(
    mapping: Mapping[str, object],
    *,
    parent_key: str = "",
    separator: str = ".",
) -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in mapping.items():
        combined_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, Mapping):
            nested = flatten_mapping(
                value,
                parent_key=combined_key,
                separator=separator,
            )
            flattened.update(nested)
        else:
            flattened[combined_key] = value
    return flattened

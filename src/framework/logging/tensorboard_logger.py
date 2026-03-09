from __future__ import annotations

import json
from collections.abc import Mapping

import torch
from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter

from .structured_logger import ExperimentLogger


class TensorBoardLogger(ExperimentLogger):
    def __init__(
        self,
        *,
        log_dir: str,
        flush_secs: int,
        max_queue: int,
    ) -> None:
        self._writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=flush_secs,
            max_queue=max_queue,
        )
        self._closed = False

    def start_run(self, *, run_name: str | None = None) -> None:
        del run_name

    def end_run(self, *, status: str = "FINISHED") -> None:
        del status
        if self._closed:
            return
        self._writer.flush()
        self._writer.close()
        self._closed = True

    def log_params(self, params: Mapping[str, object]) -> None:
        lines = [f"{key} = {format_param_value(value)}" for key, value in sorted(params.items())]
        if lines:
            self._writer.add_text("config/flattened", "\n".join(lines), 0)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        event_step = 0 if step is None else step
        for key, value in metrics.items():
            self._writer.add_scalar(key, float(value), event_step)

    def log_artifact(self, artifact_path: str) -> None:
        del artifact_path

    def log_text(self, tag: str, text: str, *, step: int | None = None) -> None:
        event_step = 0 if step is None else step
        self._writer.add_text(tag, text, event_step)

    def log_histogram(self, tag: str, values: torch.Tensor, *, step: int | None = None) -> None:
        event_step = 0 if step is None else step
        self._writer.add_histogram(tag, values, event_step)

    def log_figure(
        self,
        tag: str,
        figure: Figure | list[Figure],
        *,
        step: int | None = None,
    ) -> None:
        event_step = 0 if step is None else step
        self._writer.add_figure(tag, figure, global_step=event_step)


def format_param_value(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)

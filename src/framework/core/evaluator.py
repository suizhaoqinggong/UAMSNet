from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from framework.contracts import Batch, Metric, MetricResults, ModelAdapter, Task

from .device import autocast_context, move_batch_to_device, no_grad_context


class Evaluator:
    def __init__(
        self,
        *,
        task: Task,
        metrics: Sequence[Metric],
        device: torch.device,
        amp_enabled: bool = False,
    ) -> None:
        self._task = task
        self._metrics = list(metrics)
        self._device = device
        self._amp_enabled = amp_enabled

    def evaluate(self, *, model: ModelAdapter, batches: Iterable[Batch]) -> MetricResults:
        set_model_mode(model, training=False)
        reset_metrics(self._metrics)

        total_loss = 0.0
        batch_count = 0

        with no_grad_context():
            for batch in batches:
                device_batch = move_batch_to_device(batch, device=self._device)
                with autocast_context(enabled=self._amp_enabled, device=self._device):
                    outputs = model.forward(device_batch)
                    loss = self._task.compute_loss(outputs, device_batch)

                predictions = self._task.postprocess_outputs(outputs)
                targets = self._task.extract_targets(device_batch)
                update_metrics(self._metrics, predictions, targets)

                total_loss += to_float(loss)
                batch_count += 1

        if batch_count == 0:
            raise ValueError("Evaluator received zero batches.")

        results = collect_metric_results(self._metrics)
        results["loss"] = total_loss / batch_count
        return results


def set_model_mode(model: ModelAdapter, *, training: bool) -> None:
    model.train(mode=training)


def reset_metrics(metrics: Sequence[Metric]) -> None:
    for metric in metrics:
        metric.reset()


def update_metrics(
    metrics: Sequence[Metric],
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    for metric in metrics:
        metric.update(predictions, targets)


def collect_metric_results(metrics: Sequence[Metric]) -> MetricResults:
    aggregated: MetricResults = {}
    for metric in metrics:
        computed = metric.compute()
        for key, value in computed.items():
            if key in aggregated:
                raise ValueError(f"Duplicate metric key produced: {key}")
            aggregated[key] = value
    return aggregated


def to_float(value: torch.Tensor | int | float) -> float:
    if isinstance(value, bool):
        raise TypeError("Boolean loss values are not supported.")
    if isinstance(value, int | float):
        return float(value)
    if value.numel() != 1:
        raise ValueError("Loss tensor must be scalar.")
    return float(value.detach().item())

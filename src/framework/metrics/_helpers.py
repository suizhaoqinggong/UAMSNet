from __future__ import annotations

import torch


def prepare_metric_inputs(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return preds.detach().cpu(), targets.detach().cpu()


def scalar_metric_value(value: torch.Tensor | float) -> float:
    if isinstance(value, float):
        return value
    if value.numel() != 1:
        raise ValueError("Metric compute() must return a scalar tensor for this wrapper.")
    return float(value.item())

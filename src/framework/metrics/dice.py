"""Dice coefficient metric for segmentation."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from framework.contracts import Metric, MetricResults, Predictions, Targets


@dataclass(slots=True)
class DiceMetric(Metric):
    """Dice coefficient metric for image segmentation."""

    num_classes: int
    include_background: bool = False
    threshold: float = 0.5
    name: str = "dice"
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        self._dice_scores: list[torch.Tensor] = []

    def update(self, preds: Predictions, targets: Targets) -> None:
        """Update metric with predictions and targets."""
        # Apply threshold to get binary predictions
        binary_preds = (preds >= self.threshold).float()

        # Flatten spatial dimensions
        preds_flat = binary_preds.view(binary_preds.size(0), binary_preds.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)

        # Compute Dice per sample per class
        intersection = (preds_flat * targets_flat).sum(dim=2)
        union = preds_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)

        self._dice_scores.append(dice)

    def compute(self) -> MetricResults:
        """Compute final metric value."""
        if not self._dice_scores:
            return {self.name: 0.0}

        all_dice = torch.cat(self._dice_scores, dim=0)  # [N, C]

        # Optionally exclude background
        if not self._include_background:
            all_dice = all_dice[:, 1:]

        avg_dice = all_dice.mean().item()

        return {self.name: avg_dice}

    def reset(self) -> None:
        """Reset metric state."""
        self._dice_scores = []

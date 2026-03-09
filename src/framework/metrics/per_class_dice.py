"""Per-class Dice coefficient metric for segmentation."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from framework.contracts import Metric, MetricResults, Predictions, Targets


@dataclass(slots=True)
class PerClassDiceMetric(Metric):
    """Per-class Dice coefficient metric for image segmentation."""

    num_classes: int
    class_names: Sequence[str] | None = None
    include_background: bool = False
    threshold: float = 0.5
    name: str = "per_class_dice"
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        self._dice_scores: list[torch.Tensor] = []
        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]

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
        """Compute final metric values per class."""
        if not self._dice_scores:
            return {f"dice_{self.class_names[i]}": 0.0 for i in range(self.num_classes)}

        all_dice = torch.cat(self._dice_scores, dim=0)  # [N, C]

        # Average per class across all samples
        avg_dice_per_class = all_dice.mean(dim=0)  # [C]

        # Build results dictionary
        results = {}
        start_idx = 0 if self.include_background else 1

        for i in range(start_idx, self.num_classes):
            class_name = self.class_names[i]
            results[f"dice_{class_name}"] = avg_dice_per_class[i].item()

        # Also compute average Dice
        if not self.include_background:
            avg_dice = avg_dice_per_class[1:].mean().item()
        else:
            avg_dice = avg_dice_per_class.mean().item()
        results["avg_dice"] = avg_dice

        return results

    def reset(self) -> None:
        """Reset metric state."""
        self._dice_scores = []

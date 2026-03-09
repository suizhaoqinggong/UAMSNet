"""Segmentation task for medical image segmentation."""
from __future__ import annotations

from typing import cast

import torch
from torch import nn

from framework.contracts import (
    Batch,
    LossValue,
    ModelOutput,
    Predictions,
    ProblemType,
    Targets,
    Task,
)


class SegmentationTask(Task):
    """Segmentation task with Dice + BCE loss for medical image segmentation."""

    def __init__(
        self,
        *,
        num_classes: int,
        include_background: bool = False,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
    ) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1 for segmentation tasks.")
        if dice_weight < 0 or bce_weight < 0:
            raise ValueError("dice_weight and bce_weight must be non-negative.")
        if dice_weight + bce_weight == 0:
            raise ValueError("At least one of dice_weight or bce_weight must be positive.")

        self._num_classes = num_classes
        self._include_background = include_background
        self._dice_weight = dice_weight
        self._bce_weight = bce_weight
        self._bce_loss = nn.BCEWithLogitsLoss()

    def compute_loss(self, outputs: ModelOutput, batch: Batch) -> LossValue:
        """Compute combined Dice + BCE loss."""
        targets = self.extract_targets(batch)
        logits = outputs

        # BCE loss
        bce_loss = self._bce_loss(logits, targets.float())

        # Dice loss
        dice_loss = self._compute_dice_loss(logits, targets)

        # Combined loss
        total_loss = self._dice_weight * dice_loss + self._bce_weight * bce_loss
        return cast(LossValue, total_loss)

    def _compute_dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss."""
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)  # [B, C, H*W]
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)  # [B, C, H*W]

        # Compute Dice coefficient per sample per class
        intersection = (probs_flat * targets_flat).sum(dim=2)  # [B, C]
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # [B, C]

        # Dice coefficient
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)  # [B, C]

        # Optionally exclude background
        if not self._include_background:
            dice = dice[:, 1:]  # Exclude class 0 (background)

        # Average over classes and batch
        dice_loss = 1.0 - dice.mean()

        return dice_loss

    def extract_targets(self, batch: Batch) -> Targets:
        """Extract targets from batch."""
        return batch["label"].to(dtype=torch.float32)

    def postprocess_outputs(self, outputs: ModelOutput) -> Predictions:
        """Postprocess model outputs to predictions."""
        return torch.sigmoid(outputs)

    def infer_problem_type(self) -> ProblemType:
        """Return problem type."""
        return "multiclass_segmentation"

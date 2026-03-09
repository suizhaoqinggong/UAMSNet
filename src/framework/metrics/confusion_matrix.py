from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import MultilabelConfusionMatrix

from framework.contracts import Metric, MetricResults

from ._helpers import prepare_metric_inputs


@dataclass(slots=True)
class ConfusionMatrixMetric(Metric):
    num_classes: int
    threshold: float = 0.5
    class_names: list[str] | tuple[str, ...] | None = None
    name: str = "confusion_matrix"
    higher_is_better: bool = False
    _metric: TorchMetric = field(init=False)
    _class_labels: tuple[str, ...] = field(init=False)
    _class_keys: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("num_classes must be > 1 for multilabel confusion matrix.")
        self._metric = MultilabelConfusionMatrix(
            num_labels=self.num_classes,
            threshold=self.threshold,
        )
        self._class_labels = resolve_class_labels(
            raw_class_names=self.class_names,
            expected_size=self.num_classes,
            default_labels=tuple(f"class_{index}" for index in range(self.num_classes)),
        )
        self._class_keys = normalize_class_keys(self._class_labels)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        metric_preds, metric_targets = prepare_metric_inputs(preds, targets)
        self._metric.update(metric_preds, metric_targets)

    def compute(self) -> MetricResults:
        matrix = self._metric.compute().to(dtype=torch.float32).cpu()
        return compute_multilabel_report(
            matrix=matrix,
            class_keys=self._class_keys,
        )

    def reset(self) -> None:
        self._metric.reset()


def compute_multilabel_report(
    *,
    matrix: torch.Tensor,
    class_keys: tuple[str, ...],
) -> MetricResults:
    if matrix.dim() != 3 or matrix.shape[1:] != (2, 2):
        raise ValueError("Multilabel confusion matrix must have shape [L, 2, 2].")
    if matrix.shape[0] != len(class_keys):
        raise ValueError("class_keys length must match multilabel confusion matrix size.")

    result: MetricResults = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    specificity_values: list[float] = []
    f1_values: list[float] = []
    supports: list[float] = []

    tp_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    tn_sum = 0.0

    for index, key in enumerate(class_keys):
        tn = float(matrix[index, 0, 0].item())
        fp = float(matrix[index, 0, 1].item())
        fn = float(matrix[index, 1, 0].item())
        tp = float(matrix[index, 1, 1].item())
        support = tp + fn
        predicted = tp + fp

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        f1 = safe_f1(precision, recall)

        result[f"cm_{key}_tn"] = tn
        result[f"cm_{key}_fp"] = fp
        result[f"cm_{key}_fn"] = fn
        result[f"cm_{key}_tp"] = tp
        result[f"clinical_{key}_support"] = support
        result[f"clinical_{key}_predicted"] = predicted
        result[f"clinical_{key}_tp"] = tp
        result[f"clinical_{key}_fp"] = fp
        result[f"clinical_{key}_fn"] = fn
        result[f"clinical_{key}_tn"] = tn
        result[f"clinical_{key}_precision"] = precision
        result[f"clinical_{key}_recall"] = recall
        result[f"clinical_{key}_sensitivity"] = recall
        result[f"clinical_{key}_specificity"] = specificity
        result[f"clinical_{key}_f1"] = f1

        precision_values.append(precision)
        recall_values.append(recall)
        specificity_values.append(specificity)
        f1_values.append(f1)
        supports.append(support)

        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        tn_sum += tn

    total = tp_sum + fp_sum + fn_sum + tn_sum
    result["cm_tp"] = tp_sum
    result["cm_fp"] = fp_sum
    result["cm_fn"] = fn_sum
    result["cm_tn"] = tn_sum
    result["cm_total"] = total
    result["cm_micro_accuracy"] = safe_div(tp_sum + tn_sum, total)

    append_summary_metrics(
        result=result,
        precision_values=precision_values,
        recall_values=recall_values,
        specificity_values=specificity_values,
        f1_values=f1_values,
        supports=supports,
        tp_sum=tp_sum,
        fp_sum=fp_sum,
        fn_sum=fn_sum,
        tn_sum=tn_sum,
    )
    return result


def append_summary_metrics(
    *,
    result: MetricResults,
    precision_values: list[float],
    recall_values: list[float],
    specificity_values: list[float],
    f1_values: list[float],
    supports: list[float],
    tp_sum: float,
    fp_sum: float,
    fn_sum: float,
    tn_sum: float,
) -> None:
    macro_precision = mean_non_nan(precision_values)
    macro_recall = mean_non_nan(recall_values)
    macro_specificity = mean_non_nan(specificity_values)
    macro_f1 = mean_non_nan(f1_values)

    weighted_precision = weighted_mean_non_nan(precision_values, supports)
    weighted_recall = weighted_mean_non_nan(recall_values, supports)
    weighted_specificity = weighted_mean_non_nan(specificity_values, supports)
    weighted_f1 = weighted_mean_non_nan(f1_values, supports)

    micro_precision = safe_div(tp_sum, tp_sum + fp_sum)
    micro_recall = safe_div(tp_sum, tp_sum + fn_sum)
    micro_specificity = safe_div(tn_sum, tn_sum + fp_sum)
    micro_f1 = safe_f1(micro_precision, micro_recall)

    result["clinical_support_total"] = float(sum(supports))
    result["clinical_macro_precision"] = macro_precision
    result["clinical_macro_recall"] = macro_recall
    result["clinical_macro_sensitivity"] = macro_recall
    result["clinical_macro_specificity"] = macro_specificity
    result["clinical_macro_f1"] = macro_f1
    result["clinical_micro_precision"] = micro_precision
    result["clinical_micro_recall"] = micro_recall
    result["clinical_micro_sensitivity"] = micro_recall
    result["clinical_micro_specificity"] = micro_specificity
    result["clinical_micro_f1"] = micro_f1
    result["clinical_weighted_precision"] = weighted_precision
    result["clinical_weighted_recall"] = weighted_recall
    result["clinical_weighted_sensitivity"] = weighted_recall
    result["clinical_weighted_specificity"] = weighted_specificity
    result["clinical_weighted_f1"] = weighted_f1
    result["clinical_balanced_accuracy"] = macro_recall


def resolve_class_labels(
    *,
    raw_class_names: list[str] | tuple[str, ...] | None,
    expected_size: int,
    default_labels: tuple[str, ...],
) -> tuple[str, ...]:
    if raw_class_names is None:
        return default_labels
    if len(raw_class_names) != expected_size:
        raise ValueError("class_names length must match the confusion matrix class dimension.")
    validated: list[str] = []
    for item in raw_class_names:
        stripped = item.strip()
        if not stripped:
            raise ValueError("class_names cannot contain empty labels.")
        validated.append(stripped)
    return tuple(validated)


def normalize_class_keys(class_labels: tuple[str, ...]) -> tuple[str, ...]:
    keys: list[str] = []
    seen: dict[str, int] = {}
    for index, label in enumerate(class_labels):
        normalized = normalize_key_label(label, fallback=f"class_{index}")
        count = seen.get(normalized, 0)
        seen[normalized] = count + 1
        if count > 0:
            normalized = f"{normalized}_{count}"
        keys.append(normalized)
    return tuple(keys)


def normalize_key_label(label: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    if not normalized:
        return fallback
    return normalized


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def safe_f1(precision: float, recall: float) -> float:
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    denominator = precision + recall
    if denominator == 0.0:
        return float("nan")
    return 2.0 * precision * recall / denominator


def mean_non_nan(values: list[float]) -> float:
    finite_values = [value for value in values if not math.isnan(value)]
    if not finite_values:
        return float("nan")
    return float(sum(finite_values) / len(finite_values))


def weighted_mean_non_nan(values: list[float], weights: list[float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in zip(values, weights, strict=True):
        if math.isnan(value) or weight <= 0.0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator

from __future__ import annotations

from dataclasses import dataclass

from framework.contracts import DataAdapter, Metric, ModelAdapter, Task
from framework.metrics import (
    AccuracyMetric,
    AUCMetric,
    ConfusionMatrixMetric,
    DiceMetric,
    PerClassDiceMetric,
    PrecisionRecallF1Metric,
)

from .registry import Registry


@dataclass(frozen=True, slots=True)
class RegistryBundle:
    data_adapters: Registry[DataAdapter]
    models: Registry[ModelAdapter]
    tasks: Registry[Task]
    metrics: Registry[Metric]


def create_default_registries() -> RegistryBundle:
    data_registry = Registry[DataAdapter]("data adapter")
    model_registry = Registry[ModelAdapter]("model")
    task_registry = Registry[Task]("task")
    metric_registry = Registry[Metric]("metric")

    register_builtin_data_adapters(data_registry)
    register_builtin_models(model_registry)
    register_builtin_tasks(task_registry)
    register_builtin_metrics(metric_registry)

    return RegistryBundle(
        data_adapters=data_registry,
        models=model_registry,
        tasks=task_registry,
        metrics=metric_registry,
    )


def register_builtin_data_adapters(registry: Registry[DataAdapter]) -> None:
    # Lazy import to avoid circular dependency
    from data_adapters import WORDDataAdapter
    registry.register("word", WORDDataAdapter)


def register_builtin_models(registry: Registry[ModelAdapter]) -> None:
    # Lazy import to avoid circular dependency
    from models import register_models
    register_models(registry)


def register_builtin_tasks(registry: Registry[Task]) -> None:
    # Lazy import to avoid circular dependency
    from tasks import SegmentationTask
    registry.register("segmentation", SegmentationTask)


def register_builtin_metrics(registry: Registry[Metric]) -> None:
    registry.register("accuracy", AccuracyMetric)
    registry.register("auc", AUCMetric)
    registry.register("precision_recall_f1", PrecisionRecallF1Metric)
    registry.register("confusion_matrix", ConfusionMatrixMetric)
    registry.register("dice", DiceMetric)
    registry.register("per_class_dice", PerClassDiceMetric)

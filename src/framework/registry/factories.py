from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from framework.contracts import DataAdapter, Metric, ModelAdapter, Task

from .defaults import create_default_registries
from .registry import Registry, RegistryError, normalize_name

T = TypeVar("T")


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ComponentBundle:
    data_adapter: DataAdapter
    model: ModelAdapter
    task: Task
    metrics: list[Metric]


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    threshold: float
    num_classes: int
    class_names: tuple[str, ...]


def load_experiment_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Config file not found: {config_path}")
    with config_path.open("rb") as file_obj:
        loaded = cast(dict[str, object], tomllib.load(file_obj))
    return loaded


def load_merged_experiment_config(
    *,
    config_path: str | Path,
    model_path: str | Path,
) -> dict[str, object]:
    base_config = load_experiment_config(config_path)
    model_config = load_experiment_config(model_path)
    return deep_merge_mapping(base_config, model_config, path="")


def deep_merge_mapping(
    base: Mapping[str, object],
    override: Mapping[str, object],
    *,
    path: str,
) -> dict[str, object]:
    context = "<root>" if not path else path
    base_mapping = ensure_str_key_mapping(cast(Mapping[object, object], base), context)
    override_mapping = ensure_str_key_mapping(cast(Mapping[object, object], override), context)

    merged = dict(base_mapping)
    for key, override_value in override_mapping.items():
        next_path = key if not path else f"{path}.{key}"
        if key not in merged:
            merged[key] = override_value
            continue

        current_value = merged[key]
        if isinstance(current_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = deep_merge_mapping(current_value, override_value, path=next_path)
            continue

        if isinstance(current_value, Mapping) != isinstance(override_value, Mapping):
            raise ConfigurationError(f"Type mismatch at '{next_path}'.")

        merged[key] = override_value

    return merged


def build_component_bundle(
    config: Mapping[str, object],
    *,
    data_adapters: Registry[DataAdapter],
    models: Registry[ModelAdapter],
    tasks: Registry[Task],
    metrics: Registry[Metric],
) -> ComponentBundle:
    context = build_experiment_context(config)
    data_section = require_table(config, "data")
    model_section = require_table(config, "model")
    task_section = require_table(config, "task")
    metrics_section = require_table(config, "metrics")

    data_name = require_string(data_section, "adapter", "data")
    model_name = require_string(model_section, "name", "model")
    task_name = require_string(task_section, "name", "task")
    metric_names = resolve_metric_names(metrics_section)

    data_kwargs = section_kwargs(data_section, excluded={"adapter"})
    model_kwargs = section_kwargs(model_section, excluded={"name"})
    task_kwargs = section_kwargs(task_section, excluded={"name"})
    task_kwargs["threshold"] = context.threshold
    task_kwargs["num_classes"] = context.num_classes
    metric_kwargs_by_name = metric_kwargs(metrics_section)

    data_adapter = create_component(
        registry=data_adapters,
        component_name=data_name,
        section_name="data.adapter",
        kwargs=data_kwargs,
    )
    model = create_component(
        registry=models,
        component_name=model_name,
        section_name="model.name",
        kwargs=model_kwargs,
    )
    task = create_component(
        registry=tasks,
        component_name=task_name,
        section_name="task.name",
        kwargs=task_kwargs,
    )

    built_metrics: list[Metric] = []
    for metric_name in metric_names:
        defaults = default_metric_kwargs(
            metric_name=metric_name,
            num_classes=context.num_classes,
            threshold=context.threshold,
            class_names=context.class_names,
        )
        override_kwargs = metric_kwargs_by_name.get(normalize_name(metric_name), {})
        kwargs = defaults | override_kwargs
        metric_instance = create_component(
            registry=metrics,
            component_name=metric_name,
            section_name=f"metrics.params.{metric_name}",
            kwargs=kwargs,
        )
        built_metrics.append(metric_instance)
    if not built_metrics:
        raise ConfigurationError("metrics.params must contain at least one metric.")

    return ComponentBundle(data_adapter=data_adapter, model=model, task=task, metrics=built_metrics)


def build_default_component_bundle(config: Mapping[str, object]) -> ComponentBundle:
    registries = create_default_registries()
    return build_component_bundle(
        config,
        data_adapters=registries.data_adapters,
        models=registries.models,
        tasks=registries.tasks,
        metrics=registries.metrics,
    )


def create_component(
    *,
    registry: Registry[T],
    component_name: str,
    section_name: str,
    kwargs: Mapping[str, object],
) -> T:
    try:
        return registry.create(component_name, **dict(kwargs))
    except RegistryError as exc:
        message = f"Invalid config at '{section_name}': {exc}"
        raise ConfigurationError(message) from exc


def require_table(config: Mapping[str, object], key: str) -> dict[str, object]:
    if key not in config:
        raise ConfigurationError(f"Missing required table: [{key}]")
    value = config[key]
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Table [{key}] must be a mapping.")
    return ensure_str_key_mapping(value, f"[{key}]")


def require_string(config: Mapping[str, object], key: str, context: str) -> str:
    value = config.get(key)
    if not isinstance(value, str):
        raise ConfigurationError(f"{context}.{key} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ConfigurationError(f"{context}.{key} cannot be empty.")
    return normalized


def resolve_metric_names(metrics_section: Mapping[str, object]) -> list[str]:
    params = metrics_section.get("params")
    if params is None:
        raise ConfigurationError(
            "metrics.params.<metric_name> is required to declare metrics."
        )
    if not isinstance(params, Mapping):
        raise ConfigurationError("metrics.params must be a mapping when provided.")

    names: list[str] = []
    for key, value in params.items():
        if not isinstance(key, str):
            raise ConfigurationError("metrics.params must use string keys.")
        stripped_key = key.strip()
        if not stripped_key:
            raise ConfigurationError("metrics.params cannot contain empty metric names.")
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"metrics.params.{stripped_key} must be a mapping.")
        names.append(stripped_key)

    if not names:
        raise ConfigurationError("metrics.params is empty; define at least one metric section.")
    return names


def build_experiment_context(config: Mapping[str, object]) -> ExperimentContext:
    experiment_section = require_table(config, "experiment")
    if "num_classes" in experiment_section:
        raise ConfigurationError(
            "experiment.num_classes is not supported; use experiment.class_names only."
        )
    threshold = read_float(experiment_section, "threshold", default=0.5)
    configured_class_names = optional_string_list(experiment_section, "class_names")

    if threshold <= 0.0 or threshold >= 1.0:
        raise ConfigurationError("experiment.threshold must be in (0, 1).")
    if configured_class_names is None:
        raise ConfigurationError(
            "experiment.class_names is required for multilabel tasks."
        )
    num_classes = len(configured_class_names)
    if num_classes <= 1:
        raise ConfigurationError("experiment.class_names must define at least 2 labels.")
    class_names = tuple(configured_class_names)

    return ExperimentContext(
        threshold=threshold,
        num_classes=num_classes,
        class_names=class_names,
    )


def section_kwargs(section: Mapping[str, object], *, excluded: set[str]) -> dict[str, object]:
    return {key: section[key] for key in sorted(section) if key not in excluded}


def default_metric_kwargs(
    *,
    metric_name: str,
    num_classes: int,
    threshold: float,
    class_names: tuple[str, ...],
) -> dict[str, object]:
    normalized = metric_name.strip().lower()
    kwargs: dict[str, object] = {}

    if normalized in {"accuracy", "precision_recall_f1", "confusion_matrix"}:
        kwargs["threshold"] = threshold

    if normalized in {"accuracy", "auc", "precision_recall_f1", "confusion_matrix"}:
        kwargs["num_classes"] = num_classes

    if normalized in {"confusion_matrix"}:
        kwargs["class_names"] = class_names

    return kwargs


def metric_kwargs(metrics_section: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw_params = metrics_section.get("params")
    if raw_params is None:
        return {}
    if not isinstance(raw_params, Mapping):
        raise ConfigurationError(
            "metrics.params must be a mapping of metric-name to parameter table."
        )

    validated: dict[str, dict[str, object]] = {}
    for metric_name, raw_kwargs in ensure_str_key_mapping(raw_params, "metrics.params").items():
        if not isinstance(raw_kwargs, Mapping):
            raise ConfigurationError(f"metrics.params.{metric_name} must be a mapping.")
        global_keys = {"problem_type", "num_classes", "threshold", "class_names"}
        invalid_keys = sorted(global_keys.intersection(raw_kwargs.keys()))
        if invalid_keys:
            raise ConfigurationError(
                f"metrics.params.{metric_name} contains disallowed keys: {', '.join(invalid_keys)}."
            )
        validated[normalize_name(metric_name)] = section_kwargs(
            ensure_str_key_mapping(raw_kwargs, f"metrics.params.{metric_name}"),
            excluded=set(),
        )
    return validated


def optional_int(config: Mapping[str, object], key: str, *, minimum: int) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{key} must be an integer when provided.")
    if value < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum}.")
    return value


def optional_string_list(config: Mapping[str, object], key: str) -> list[str] | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ConfigurationError(f"{key} must be a list of strings when provided.")

    validated: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ConfigurationError(f"{key} must contain only strings.")
        stripped = item.strip()
        if not stripped:
            raise ConfigurationError(f"{key} cannot contain empty entries.")
        validated.append(stripped)

    if not validated:
        raise ConfigurationError(f"{key} cannot be empty when provided.")
    return validated


def read_float(
    config: Mapping[str, object],
    key: str,
    *,
    default: float,
    minimum: float | None = None,
) -> float:
    value = config.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigurationError(f"{key} must be numeric.")
    parsed = float(value)
    if minimum is not None and parsed < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum}.")
    return parsed


def ensure_str_key_mapping(mapping: Mapping[object, object], context: str) -> dict[str, object]:
    validated: dict[str, object] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise ConfigurationError(f"{context} must use string keys only.")
        validated[key] = value
    return validated

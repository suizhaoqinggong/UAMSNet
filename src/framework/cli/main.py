from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, cast

import torch
from torch.utils.data import DataLoader

from framework.contracts import (
    Batch,
    DataAdapter,
    DatasetLike,
    Metric,
    MetricResults,
    ModelAdapter,
)

T = TypeVar("T")
from framework.core import (
    DEFAULT_TENSORBOARD_SCALAR_TAGS,
    LayoutConfig,
    OutputSettings,
    RunLayout,
    Trainer,
    TrainerSettings,
    resolve_run_layout,
)
from framework.registry import (
    ConfigurationError,
    Registry,
    create_default_registries,
    load_merged_experiment_config,
)


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    threshold: float
    num_classes: int
    class_names: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="framework")
    parser.add_argument(
        "command",
        choices=["train", "validate", "test", "fit"],
        help="Framework command to run.",
    )
    parser.add_argument("--configs", required=True, help="Path to shared TOML config.")
    parser.add_argument("--model", required=True, help="Path to TOML model declaration.")
    parser.add_argument("--checkpoint", help="Path to checkpoint for validate/test.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"validate", "test"} and not args.checkpoint:
        parser.error("--checkpoint is required for validate and test commands.")

    config = load_merged_experiment_config(
        config_path=args.configs,
        model_path=args.model,
    )
    results = run_command(
        command=args.command,
        config=config,
        config_path=args.configs,
        model_path=args.model,
        checkpoint_override=args.checkpoint,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def run_command(
    *,
    command: str,
    config: Mapping[str, object],
    config_path: str,
    model_path: str,
    checkpoint_override: str | None,
) -> MetricResults:
    registries = create_default_registries()
    context = build_experiment_context(config)
    output_settings = build_output_settings(config)
    run_layout = resolve_run_layout(
        LayoutConfig(
            experiment_config_path=Path(config_path),
            model_config_path=Path(model_path),
            output=output_settings,
        )
    )

    data_section = require_table(config, "data")
    model_section = require_table(config, "model")
    task_section = require_table(config, "task")

    data_adapter_name = require_string(data_section, "adapter", "data")
    data_kwargs = extract_component_kwargs(
        data_section,
        name_key="adapter",
        reserved={"batch_size", "num_workers", "pin_memory", "persistent_workers"},
    )
    data_kwargs = apply_global_context(
        registry=registries.data_adapters,
        component_name=data_adapter_name,
        kwargs=data_kwargs,
        context=context,
        include_threshold=False,
    )
    data_adapter = registries.data_adapters.create(data_adapter_name, **data_kwargs)

    model_name = require_string(model_section, "name", "model")
    model_kwargs = extract_component_kwargs(model_section, name_key="name", reserved=set())
    model_kwargs = apply_global_context(
        registry=registries.models,
        component_name=model_name,
        kwargs=model_kwargs,
        context=context,
        include_threshold=False,
    )
    model = registries.models.create(model_name, **model_kwargs)

    task_name = require_string(task_section, "name", "task")
    task_kwargs = extract_component_kwargs(task_section, name_key="name", reserved=set())
    task_kwargs = apply_global_context(
        registry=registries.tasks,
        component_name=task_name,
        kwargs=task_kwargs,
        context=context,
        include_threshold=True,
    )
    task = registries.tasks.create(task_name, **task_kwargs)

    data_adapter.prepare()
    train_dataset, val_dataset, test_dataset = data_adapter.get_splits()

    train_loader, val_loader, test_loader = build_dataloaders(
        data_section=data_section,
        data_adapter=data_adapter,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    metrics_section = require_table(config, "metrics")
    metric_names = resolve_metric_names(metrics_section)
    train_metrics = build_metric_list(
        metric_names=metric_names,
        metrics_section=metrics_section,
        registry=registries.metrics,
        context=context,
    )
    val_metrics = build_metric_list(
        metric_names=metric_names,
        metrics_section=metrics_section,
        registry=registries.metrics,
        context=context,
    )
    test_metrics = build_metric_list(
        metric_names=metric_names,
        metrics_section=metrics_section,
        registry=registries.metrics,
        context=context,
    )

    optimizer = build_optimizer(model=model, config=config)
    trainer_settings = build_trainer_settings(config=config, run_layout=run_layout)
    trainer = Trainer(
        model=model,
        task=task,
        optimizer=optimizer,
        train_batches=train_loader,
        val_batches=val_loader,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        settings=trainer_settings,
        test_batches=test_loader,
        test_metrics=test_metrics,
        config_snapshot=config,
    )

    if command == "train":
        return trainer.train()
    if command == "validate":
        if checkpoint_override is None:
            raise ConfigurationError("validate command requires a checkpoint path.")
        return trainer.validate(checkpoint_path=checkpoint_override)
    if command == "test":
        if checkpoint_override is None:
            raise ConfigurationError("test command requires a checkpoint path.")
        return trainer.test(checkpoint_path=checkpoint_override)
    if command == "fit":
        return trainer.fit()
    raise ConfigurationError(f"Unsupported command: {command}")


def build_dataloaders(
    *,
    data_section: Mapping[str, object],
    data_adapter: DataAdapter,
    train_dataset: DatasetLike,
    val_dataset: DatasetLike,
    test_dataset: DatasetLike,
) -> tuple[Iterable[Batch], Iterable[Batch], Iterable[Batch]]:
    batch_size = read_int(data_section, "batch_size", default=64, minimum=1)
    num_workers = read_int(data_section, "num_workers", default=0, minimum=0)
    pin_memory = read_bool(data_section, "pin_memory", default=torch.cuda.is_available())
    persistent_workers = read_bool(
        data_section,
        "persistent_workers",
        default=num_workers > 0,
    )

    if num_workers > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=data_adapter.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=data_adapter.collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=data_adapter.collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=data_adapter.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=data_adapter.collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=data_adapter.collate_fn,
        )
    return (
        cast(Iterable[Batch], train_loader),
        cast(Iterable[Batch], val_loader),
        cast(Iterable[Batch], test_loader),
    )


def build_metric_list(
    *,
    metric_names: list[str],
    metrics_section: Mapping[str, object],
    registry: Registry[Metric],
    context: ExperimentContext,
) -> list[Metric]:
    metric_params = optional_nested_table(metrics_section, "params")

    created: list[Metric] = []
    for metric_name in metric_names:
        defaults = default_metric_kwargs(
            metric_name=metric_name,
            num_classes=context.num_classes,
            threshold=context.threshold,
            class_names=context.class_names,
        )
        override = metric_params.get(metric_name, {})
        if not isinstance(override, Mapping):
            raise ConfigurationError(f"metrics.params.{metric_name} must be a mapping.")
        for disallowed_key in ("problem_type", "num_classes", "threshold", "class_names"):
            if disallowed_key in override:
                raise ConfigurationError(
                    f"metrics.params.{metric_name}.{disallowed_key} is not allowed."
                )
        kwargs = {**defaults, **dict(override)}
        created_metric = registry.create(metric_name, **kwargs)
        created.append(created_metric)

    return created


def resolve_metric_names(metrics_section: Mapping[str, object]) -> list[str]:
    params_value = metrics_section.get("params")
    if params_value is None:
        raise ConfigurationError(
            "metrics.params.<metric_name> is required to declare metrics."
        )
    if not isinstance(params_value, Mapping):
        raise ConfigurationError("metrics.params must be a mapping when provided.")

    names: list[str] = []
    for key, value in params_value.items():
        if not isinstance(key, str):
            raise ConfigurationError("metrics.params must use string keys.")
        stripped_key = key.strip()
        if not stripped_key:
            raise ConfigurationError("metrics.params cannot contain empty metric names.")
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"metrics.params.{stripped_key} must be a mapping.")
        names.append(stripped_key)

    if not names:
        raise ConfigurationError(
            "metrics.params is empty; define at least one metric configuration."
        )
    return names


def build_optimizer(*, model: ModelAdapter, config: Mapping[str, object]) -> torch.optim.Optimizer:
    optimizer_section = optional_table(config, "optimizer")
    name = require_string(optimizer_section, "name", "optimizer", default="adam").lower()
    lr = read_float(optimizer_section, "lr", default=1e-3, minimum=1e-12)
    weight_decay = read_float(optimizer_section, "weight_decay", default=0.0, minimum=0.0)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = read_float(optimizer_section, "momentum", default=0.9, minimum=0.0)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ConfigurationError(f"Unsupported optimizer.name: {name}")


def build_trainer_settings(
    *,
    config: Mapping[str, object],
    run_layout: RunLayout,
) -> TrainerSettings:
    trainer_section = optional_table(config, "trainer")
    experiment_section = optional_table(config, "experiment")
    checkpoint_section = optional_table(config, "checkpoint")
    logging_section = optional_table(config, "logging")
    tensorboard_section = optional_table(logging_section, "tensorboard")

    epochs = read_int(trainer_section, "epochs", default=1, minimum=1)
    device = require_string(trainer_section, "device", "trainer", default="auto")
    amp = read_bool(trainer_section, "amp", default=False)
    grad_clip_norm = optional_float(trainer_section, "grad_clip_norm", minimum=0.0)
    deterministic = read_bool(trainer_section, "deterministic", default=True)
    log_interval_steps = read_int(trainer_section, "log_interval_steps", default=20, minimum=1)
    patience = optional_int(trainer_section, "patience", minimum=1)

    trainer_seed = optional_int(trainer_section, "seed", minimum=0)
    experiment_seed = optional_int(experiment_section, "seed", minimum=0)
    seed = trainer_seed if trainer_seed is not None else experiment_seed

    if "directory" in checkpoint_section:
        raise ConfigurationError("checkpoint.directory is not supported.")
    checkpoint_monitor = require_string(
        checkpoint_section,
        "monitor",
        "checkpoint",
        default="loss",
    )
    checkpoint_mode = require_string(
        checkpoint_section,
        "mode",
        "checkpoint",
        default="min",
    )
    if checkpoint_mode not in {"min", "max"}:
        raise ConfigurationError("checkpoint.mode must be either 'min' or 'max'.")

    tensorboard_enabled = read_bool(tensorboard_section, "enabled", default=True)
    tensorboard_flush_secs = read_int(tensorboard_section, "flush_secs", default=30, minimum=1)
    tensorboard_max_queue = read_int(tensorboard_section, "max_queue", default=10, minimum=1)
    histogram_every_n_epochs = read_int(
        tensorboard_section,
        "histogram_every_n_epochs",
        default=1,
        minimum=1,
    )
    confusion_matrix_every_n_epochs = read_int(
        tensorboard_section,
        "confusion_matrix_every_n_epochs",
        default=1,
        minimum=1,
    )
    tensorboard_scalar_tags = optional_string_list(tensorboard_section, "scalar_tags")

    resume_from = optional_string(trainer_section, "resume_from")

    return TrainerSettings(
        epochs=epochs,
        run_dir=str(run_layout.run_dir),
        checkpoints_dir=str(run_layout.checkpoints_dir),
        tensorboard_dir=str(run_layout.tensorboard_dir),
        logs_dir=str(run_layout.logs_dir),
        confusion_matrix_dir=str(run_layout.confusion_matrix_dir),
        tensorboard_enabled=tensorboard_enabled,
        tensorboard_flush_secs=tensorboard_flush_secs,
        tensorboard_max_queue=tensorboard_max_queue,
        histogram_every_n_epochs=histogram_every_n_epochs,
        confusion_matrix_every_n_epochs=confusion_matrix_every_n_epochs,
        tensorboard_scalar_tags=(
            tuple(tensorboard_scalar_tags)
            if tensorboard_scalar_tags is not None
            else DEFAULT_TENSORBOARD_SCALAR_TAGS
        ),
        log_interval_steps=log_interval_steps,
        patience=patience,
        device=device,
        amp=amp,
        grad_clip_norm=grad_clip_norm,
        seed=seed,
        deterministic=deterministic,
        checkpoint_monitor=checkpoint_monitor,
        checkpoint_mode=cast(Literal["min", "max"], checkpoint_mode),
        resume_from=resume_from,
    )


def build_output_settings(config: Mapping[str, object]) -> OutputSettings:
    output_section = optional_table(config, "output")
    root_dir = require_string(output_section, "root_dir", "output", default="artifacts/runs")
    backup_code = read_bool(output_section, "backup_code", default=True)
    return OutputSettings(root_dir=Path(root_dir), backup_code=backup_code)


def extract_component_kwargs(
    section: Mapping[str, object],
    *,
    name_key: str,
    reserved: set[str],
) -> dict[str, object]:
    explicit_params = section.get("params")
    if explicit_params is not None:
        if not isinstance(explicit_params, Mapping):
            raise ConfigurationError(f"{name_key} section params must be a mapping.")
        return {str(key): value for key, value in explicit_params.items()}

    return {
        key: value
        for key, value in section.items()
        if key not in reserved and key not in {name_key, "params"}
    }


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


def apply_global_context(
    *,
    registry: Registry[T],
    component_name: str,
    kwargs: dict[str, object],
    context: ExperimentContext,
    include_threshold: bool,
) -> dict[str, object]:
    candidates: dict[str, object] = {"num_classes": context.num_classes}
    if include_threshold:
        candidates["threshold"] = context.threshold

    return inject_supported_kwargs(
        factory=registry.get(component_name),
        kwargs=kwargs,
        candidates=candidates,
    )


def inject_supported_kwargs(
    *,
    factory: Callable[..., object],
    kwargs: dict[str, object],
    candidates: Mapping[str, object],
) -> dict[str, object]:
    resolved = dict(kwargs)

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return resolved

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    accepted_names = {
        parameter_name
        for parameter_name, parameter in signature.parameters.items()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }

    for key, value in candidates.items():
        if accepts_var_kwargs or key in accepted_names:
            resolved[key] = value

    return resolved


def require_table(config: Mapping[str, object], key: str) -> dict[str, object]:
    if key not in config:
        raise ConfigurationError(f"Missing required table: [{key}]")
    value = config[key]
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Table [{key}] must be a mapping.")
    return {str(inner_key): inner_value for inner_key, inner_value in value.items()}


def optional_table(config: Mapping[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Table [{key}] must be a mapping.")
    return {str(inner_key): inner_value for inner_key, inner_value in value.items()}


def optional_nested_table(section: Mapping[str, object], key: str) -> dict[str, dict[str, object]]:
    value = section.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Section key '{key}' must be a mapping.")

    nested: dict[str, dict[str, object]] = {}
    for nested_key, nested_value in value.items():
        if not isinstance(nested_key, str):
            raise ConfigurationError(f"Section key '{key}' must use string keys.")
        if not isinstance(nested_value, Mapping):
            raise ConfigurationError(f"Section key '{key}.{nested_key}' must be a mapping.")
        nested[nested_key] = {
            str(inner_key): inner_value
            for inner_key, inner_value in nested_value.items()
        }
    return nested


def require_string(
    section: Mapping[str, object],
    key: str,
    context: str,
    *,
    default: str | None = None,
) -> str:
    value = section.get(key, default)
    if not isinstance(value, str):
        raise ConfigurationError(f"{context}.{key} must be a string.")
    stripped = value.strip()
    if not stripped:
        raise ConfigurationError(f"{context}.{key} cannot be empty.")
    return stripped


def read_int(
    section: Mapping[str, object],
    key: str,
    *,
    default: int,
    minimum: int,
) -> int:
    value = section.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{key} must be an integer.")
    if value < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum}.")
    return value


def optional_int(section: Mapping[str, object], key: str, *, minimum: int) -> int | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{key} must be an integer when provided.")
    if value < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum}.")
    return value


def read_float(
    section: Mapping[str, object],
    key: str,
    *,
    default: float,
    minimum: float | None = None,
) -> float:
    value = section.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigurationError(f"{key} must be numeric.")
    parsed = float(value)
    if minimum is not None and parsed < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum}.")
    return parsed


def optional_float(section: Mapping[str, object], key: str, *, minimum: float) -> float | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigurationError(f"{key} must be numeric when provided.")
    parsed = float(value)
    if parsed < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum}.")
    return parsed


def read_bool(section: Mapping[str, object], key: str, *, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{key} must be a boolean.")
    return value


def optional_string(section: Mapping[str, object], key: str) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigurationError(f"{key} must be a string when provided.")
    stripped = value.strip()
    return stripped or None


def optional_string_list(section: Mapping[str, object], key: str) -> list[str] | None:
    value = section.get(key)
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


if __name__ == "__main__":
    raise SystemExit(main())

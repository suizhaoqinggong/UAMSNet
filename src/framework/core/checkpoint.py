from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch


class CheckpointError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CheckpointSettings:
    directory: str
    monitor: str
    mode: Literal["min", "max"] = "max"
    best_filename: str = "best.ckpt"
    last_filename: str = "last.ckpt"
    save_last: bool = True

    def __post_init__(self) -> None:
        if not self.directory.strip():
            raise ValueError("checkpoint directory cannot be empty.")
        if not self.monitor.strip():
            raise ValueError("checkpoint monitor cannot be empty.")


class CheckpointManager:
    def __init__(
        self,
        settings: CheckpointSettings,
        *,
        best_monitor_value: float | None = None,
        best_epoch: int | None = None,
    ) -> None:
        self._settings = settings
        self._directory = Path(settings.directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._best_monitor_value = best_monitor_value
        self._best_epoch = best_epoch

    @property
    def settings(self) -> CheckpointSettings:
        return self._settings

    @property
    def best_path(self) -> Path:
        return self._directory / self._settings.best_filename

    @property
    def last_path(self) -> Path:
        return self._directory / self._settings.last_filename

    @property
    def best_monitor_value(self) -> float | None:
        return self._best_monitor_value

    @property
    def best_epoch(self) -> int | None:
        return self._best_epoch

    def update_best_state(self, *, best_monitor_value: float, best_epoch: int) -> None:
        self._best_monitor_value = best_monitor_value
        self._best_epoch = best_epoch

    def maybe_save(
        self,
        *,
        epoch: int,
        train_metrics: Mapping[str, float],
        val_metrics: Mapping[str, float],
        model_state: Mapping[str, object],
        optimizer_state: Mapping[str, object],
        scaler_state: Mapping[str, object] | None,
        config_snapshot: Mapping[str, object] | None,
    ) -> bool:
        monitor_value = require_float_metric(val_metrics, self._settings.monitor)
        payload = build_checkpoint_payload(
            epoch=epoch,
            monitor_metric=self._settings.monitor,
            monitor_mode=self._settings.mode,
            monitor_value=monitor_value,
            best_monitor_value=self._best_monitor_value,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scaler_state=scaler_state,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            config_snapshot=config_snapshot,
        )

        if self._settings.save_last:
            save_checkpoint_payload(payload, self.last_path)

        improved = is_improved(
            current=monitor_value,
            best=self._best_monitor_value,
            mode=self._settings.mode,
        )
        if improved:
            self._best_monitor_value = monitor_value
            self._best_epoch = epoch
            payload["best_monitor_value"] = monitor_value
            save_checkpoint_payload(payload, self.best_path)
        return improved


def is_improved(*, current: float, best: float | None, mode: Literal["min", "max"]) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best
    return current < best


def build_checkpoint_payload(
    *,
    epoch: int,
    monitor_metric: str,
    monitor_mode: Literal["min", "max"],
    monitor_value: float,
    best_monitor_value: float | None,
    model_state: Mapping[str, object],
    optimizer_state: Mapping[str, object],
    scaler_state: Mapping[str, object] | None,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    config_snapshot: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "monitor_value": monitor_value,
        "best_monitor_value": best_monitor_value,
        "model_state": dict(model_state),
        "optimizer_state": dict(optimizer_state),
        "scaler_state": None if scaler_state is None else dict(scaler_state),
        "metrics": {
            "train": dict(train_metrics),
            "val": dict(val_metrics),
        },
        "config": {} if config_snapshot is None else dict(config_snapshot),
    }


def save_checkpoint_payload(payload: Mapping[str, object], path: str | Path) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), str(save_path))


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise CheckpointError(f"Checkpoint file not found: {checkpoint_path}")

    loaded = torch.load(str(checkpoint_path), map_location=map_location)
    if not isinstance(loaded, Mapping):
        raise CheckpointError("Checkpoint payload must be a mapping.")
    return ensure_mapping_object(loaded, context="checkpoint payload")


def require_float_metric(metrics: Mapping[str, float], key: str) -> float:
    if key not in metrics:
        available = ", ".join(sorted(metrics)) if metrics else "<none>"
        raise CheckpointError(f"Monitor metric '{key}' not found. Available: {available}")
    value = metrics[key]
    if isinstance(value, bool):
        raise CheckpointError(f"Metric '{key}' must be numeric.")
    return float(value)


def ensure_mapping_object(mapping: Mapping[object, object], *, context: str) -> dict[str, object]:
    validated: dict[str, object] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise CheckpointError(f"{context} must contain only string keys.")
        validated[key] = value
    return validated


def extract_epoch(payload: Mapping[str, object]) -> int:
    raw_epoch = payload.get("epoch")
    if not isinstance(raw_epoch, int) or raw_epoch < 0:
        raise CheckpointError("Checkpoint field 'epoch' must be a non-negative integer.")
    return raw_epoch


def extract_best_monitor_value(payload: Mapping[str, object]) -> float | None:
    raw_value = payload.get("best_monitor_value")
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise CheckpointError("Checkpoint field 'best_monitor_value' must be numeric or null.")
    if isinstance(raw_value, int | float):
        return float(raw_value)
    raise CheckpointError("Checkpoint field 'best_monitor_value' must be numeric or null.")


def extract_state_mapping(payload: Mapping[str, object], key: str) -> dict[str, object]:
    raw_value = payload.get(key)
    if not isinstance(raw_value, Mapping):
        raise CheckpointError(f"Checkpoint field '{key}' must be a mapping.")
    return ensure_mapping_object(cast(Mapping[object, object], raw_value), context=f"'{key}'")


def extract_optional_state_mapping(
    payload: Mapping[str, object],
    key: str,
) -> dict[str, object] | None:
    raw_value = payload.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, Mapping):
        raise CheckpointError(f"Checkpoint field '{key}' must be a mapping when present.")
    return ensure_mapping_object(cast(Mapping[object, object], raw_value), context=f"'{key}'")

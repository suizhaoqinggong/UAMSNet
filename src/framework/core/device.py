from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any, cast

import torch
from torch.cuda.amp import GradScaler

from framework.contracts import Batch


def resolve_device(requested: str | None = None) -> torch.device:
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in the runtime.")
        return torch.device(normalized)
    if normalized in {"mps", "mps:0"}:
        if not is_mps_available():
            raise RuntimeError("MPS was requested but is unavailable in the runtime.")
        return torch.device("mps")
    raise ValueError(f"Unsupported device option: {requested!r}")


def is_mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def autocast_context(enabled: bool, device: torch.device) -> AbstractContextManager[object]:
    if not enabled:
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, enabled=True)


def no_grad_context() -> AbstractContextManager[object]:
    return torch.no_grad()


def build_grad_scaler(enabled: bool, device: torch.device) -> GradScaler | None:
    if not enabled or device.type != "cuda":
        return None
    return GradScaler(enabled=True)


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    image = move_value_to_device(batch["image"], device=device)
    label = move_value_to_device(batch["label"], device=device)
    sample_ids = move_value_to_device(batch["id"], device=device)

    if not isinstance(image, torch.Tensor):
        raise TypeError("batch['image'] must be a torch.Tensor.")
    if not isinstance(label, torch.Tensor):
        raise TypeError("batch['label'] must be a torch.Tensor.")
    if not isinstance(sample_ids, torch.Tensor | list):
        raise TypeError("batch['id'] must be a torch.Tensor or list[str].")
    if isinstance(sample_ids, list) and not all(isinstance(item, str) for item in sample_ids):
        raise TypeError("batch['id'] list must contain only strings.")

    moved: Batch = {
        "image": image,
        "label": label,
        "id": sample_ids,
    }
    if "meta" in batch:
        meta = move_value_to_device(batch["meta"], device=device)
        if not isinstance(meta, Mapping):
            raise TypeError("batch['meta'] must be a mapping when present.")
        moved["meta"] = cast(Mapping[str, Any], meta)
    return moved


def move_value_to_device(value: object, *, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, list):
        return [move_value_to_device(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_value_to_device(item, device=device) for item in value)
    if isinstance(value, Mapping):
        return {key: move_value_to_device(item, device=device) for key, item in value.items()}
    return value

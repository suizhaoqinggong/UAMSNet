"""Model catalog for registering segmentation models."""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

from framework.contracts import ModelAdapter
from framework.registry.registry import Registry

from .attention_unet import AttentionUNet
from .dscnet import DSCNet
from .uamsnet import UAMSNet
from .unet import UNet

ModelFactory: TypeAlias = Callable[..., ModelAdapter]

MODEL_REGISTRY_TABLE: dict[str, ModelFactory] = {
    "uamsnet": cast(ModelFactory, UAMSNet),
    "unet": cast(ModelFactory, UNet),
    "attention_unet": cast(ModelFactory, AttentionUNet),
    "dscnet": cast(ModelFactory, DSCNet),
}


def register_models(registry: Registry[ModelAdapter]) -> None:
    """Register all models to the registry."""
    for model_name, factory in MODEL_REGISTRY_TABLE.items():
        registry.register(model_name, factory)

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .registry import GFMRegistry, get_default_registry


def create_model(
    name: str,
    *,
    seed: int = 0,
    device: str = "cuda:0",
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    registry: GFMRegistry | None = None,
    **hyperparams: Any,
):
    reg = registry or get_default_registry()
    config = ExperimentConfig(model_name=reg.canonical_name(name), seed=seed, device=device)
    if data_root is not None:
        config.data_root = Path(data_root)
    if output_root is not None:
        config.output_root = Path(output_root)
    if checkpoint_root is not None:
        config.checkpoint_root = Path(checkpoint_root)
    config.hyperparams.update(hyperparams)
    return reg.create(name, config)


def list_models(registry: GFMRegistry | None = None) -> list[str]:
    return (registry or get_default_registry()).list_models()


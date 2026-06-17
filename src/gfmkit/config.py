from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._paths import (
    DEFAULT_BASELINES_ROOT,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_LOG_ROOT,
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
)


@dataclass(slots=True)
class ExperimentConfig:
    """Runtime configuration shared by all GFM adapters."""

    model_name: str
    seed: int = 0
    device: str = "cuda:0"
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    log_root: Path = DEFAULT_LOG_ROOT
    repo_root: Path = REPO_ROOT
    baselines_root: Path = DEFAULT_BASELINES_ROOT
    hyperparams: dict[str, Any] = field(default_factory=dict)

    def with_hyperparams(self, **updates: Any) -> "ExperimentConfig":
        params = dict(self.hyperparams)
        params.update(updates)
        return ExperimentConfig(
            model_name=self.model_name,
            seed=self.seed,
            device=self.device,
            data_root=Path(self.data_root),
            output_root=Path(self.output_root),
            checkpoint_root=Path(self.checkpoint_root),
            log_root=Path(self.log_root),
            repo_root=Path(self.repo_root),
            baselines_root=Path(self.baselines_root),
            hyperparams=params,
        )

    def hparam(self, name: str, default: Any = None) -> Any:
        return self.hyperparams.get(name, default)

    @property
    def model_output_dir(self) -> Path:
        return Path(self.output_root) / self.model_name

    @property
    def model_checkpoint_dir(self) -> Path:
        return Path(self.checkpoint_root) / self.model_name

    @property
    def model_log_dir(self) -> Path:
        return Path(self.log_root) / self.model_name


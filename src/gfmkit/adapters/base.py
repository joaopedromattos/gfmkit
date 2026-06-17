from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import torch

from gfmkit.config import ExperimentConfig
from gfmkit.data import DatasetSuite, LinkPredictionDataset, load_link_prediction_splits
from gfmkit.outputs import EmbeddingOutput, save_embedding_output

from .utils import resolve_device, seed_everything

if TYPE_CHECKING:
    from torch_geometric.data import Data


class GFMModel(ABC):
    """Base class for all Graph Foundation Model adapters."""

    supports_in_process: bool = True

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.name = config.model_name
        self.seed = config.seed
        self.device = resolve_device(config.device)
        seed_everything(self.seed)

    def hparam(self, name: str, default=None):
        return self.config.hparam(name, default)

    def resolve_datasets(self, datasets: str | Iterable[str]) -> list[str]:
        return DatasetSuite().resolve(datasets)

    def load_split(self, name: str, split: str = "train", **kwargs) -> "Data":
        split_idx = {"train": 0, "val": 1, "valid": 1, "test": 2}[split]
        return LinkPredictionDataset(name, base_dir=self.config.data_root, **kwargs).splits()[split_idx]

    def load_splits(self, datasets: str | Iterable[str], split: str = "train", **kwargs) -> dict[str, "Data"]:
        return load_link_prediction_splits(
            self.resolve_datasets(datasets),
            base_dir=self.config.data_root,
            split=split,
            **kwargs,
        )

    def ensure_dirs(self) -> None:
        self.config.model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.model_log_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def pretrain(self, datasets: str | Iterable[str] = "link1"):
        raise NotImplementedError

    def finetune(self, datasets: str | Iterable[str] = "link2"):
        return None

    @abstractmethod
    def infer(self, datasets: str | Iterable[str] = "link2") -> dict[str, EmbeddingOutput | dict]:
        raise NotImplementedError

    def embed(self, data: "Data") -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not expose direct embedding inference")

    def save_outputs(
        self,
        outputs: dict[str, EmbeddingOutput | dict],
        output_dir: str | Path | None = None,
        *,
        include_metadata: bool = False,
    ) -> dict[str, Path]:
        out_dir = Path(output_dir) if output_dir is not None else self.config.model_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = {}
        for dataset, output in outputs.items():
            if isinstance(output, EmbeddingOutput):
                path = out_dir / f"{dataset}_{self.seed}.pt"
                torch.save(output.to_dict(include_metadata=include_metadata), path)
            else:
                path = out_dir / f"{dataset}_{self.seed}.pt"
                torch.save(output, path)
            saved[dataset] = path
        return saved

    def save_output(self, output: EmbeddingOutput, output_dir: str | Path | None = None) -> Path:
        return save_embedding_output(output, output_dir or self.config.model_output_dir)

    def checkpoint_path(self, name: str) -> Path:
        self.config.model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.config.model_checkpoint_dir / name

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        raise NotImplementedError(f"{self.name} does not implement save_checkpoint")

    def load_checkpoint(self, path: str | Path):
        raise NotImplementedError(f"{self.name} does not implement load_checkpoint")

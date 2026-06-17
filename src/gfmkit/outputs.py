from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class EmbeddingOutput:
    """Standard link-prediction embedding artifact."""

    z: torch.Tensor
    edge_label: torch.Tensor
    edge_label_index: torch.Tensor
    dataset: str
    model_name: str
    seed: int

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "z": self.z,
            "edge_label": self.edge_label,
            "edge_label_index": self.edge_label_index,
        }
        if include_metadata:
            payload.update(
                {
                    "dataset": self.dataset,
                    "model_name": self.model_name,
                    "seed": self.seed,
                }
            )
        return payload


def save_embedding_output(output: EmbeddingOutput, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{output.dataset}_{output.seed}.pt"
    torch.save(output.to_dict(), path)
    return path


def load_embedding_output(path: str | Path) -> dict[str, object]:
    return torch.load(Path(path), map_location="cpu")


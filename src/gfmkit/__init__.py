"""Graph Foundation Model toolkit for the ZSLP benchmark codebase."""

from .api import create_model, list_models
from .config import ExperimentConfig
from .data import DatasetSuite
from .outputs import EmbeddingOutput

__all__ = [
    "DatasetSuite",
    "EmbeddingOutput",
    "ExperimentConfig",
    "create_model",
    "list_models",
]


def __getattr__(name: str):
    if name == "LinkPredictionDataset":
        from .data import LinkPredictionDataset

        return LinkPredictionDataset
    raise AttributeError(name)

from __future__ import annotations

import pickle
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix

from ._paths import DEFAULT_DATA_ROOT
from .presets import DATASET_PRESETS, LINK1_DATASETS, LINK2_DATASETS, SMOKE_DATASETS

if TYPE_CHECKING:
    from torch_geometric.data import Data


@contextmanager
def temp_seed(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@dataclass(frozen=True, slots=True)
class DatasetSuite:
    """Canonical dataset groups used by the transfer link-prediction benchmark."""

    link1: tuple[str, ...] = LINK1_DATASETS
    link2: tuple[str, ...] = LINK2_DATASETS
    smoke: tuple[str, ...] = SMOKE_DATASETS

    @classmethod
    def from_preset(cls, preset: str = "link_prediction") -> "DatasetSuite":
        if preset not in {"link_prediction", "lp", "default"}:
            raise ValueError(f"Unknown dataset-suite preset: {preset}")
        return cls()

    def resolve(self, selector: str | Iterable[str]) -> list[str]:
        if isinstance(selector, str):
            if "," in selector:
                return [item.strip() for item in selector.split(",") if item.strip()]
            if selector in DATASET_PRESETS:
                return list(DATASET_PRESETS[selector])
            return [selector]
        return list(selector)


class LinkPredictionDataset:
    """Load a zero-shot link-prediction dataset as PyG train/val/test splits."""

    def __init__(
        self,
        name: str,
        *,
        base_dir: str | Path = DEFAULT_DATA_ROOT,
        use_fewshot: bool = False,
        fewshot_ratio: float | int = 0.1,
        add_self_loops: bool = True,
        feature_mode: str = "constant",
        constant_feature_dim: int = 128,
        split_seed: int = 42,
    ) -> None:
        self.name = name
        self.root = Path(base_dir) / name
        self.use_fewshot = use_fewshot
        self.fewshot_ratio = fewshot_ratio
        self.add_self_loops = add_self_loops
        self.feature_mode = feature_mode
        self.constant_feature_dim = constant_feature_dim
        self.split_seed = split_seed

        self.feat_file = self.root / "feats.pkl"
        self.train_file = self.root / "trn_mat.pkl"
        self.val_file = self.root / "val_mat.pkl"
        self.test_file = self.root / "tst_mat.pkl"

    @staticmethod
    def _load_coo(path: Path) -> coo_matrix:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset matrix: {path}")
        with path.open("rb") as handle:
            matrix = (pickle.load(handle) != 0).astype(np.float32)
        if not sp.isspmatrix_coo(matrix):
            matrix = sp.coo_matrix(matrix)
        return matrix

    @staticmethod
    def _load_feats(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _normalize_adj(matrix: coo_matrix) -> coo_matrix:
        degree_r = np.asarray(matrix.sum(axis=-1)).flatten()
        d_inv_sqrt_r = np.power(degree_r, -0.5, where=degree_r > 0)
        d_inv_sqrt_r[~np.isfinite(d_inv_sqrt_r)] = 0.0
        d_r = sp.diags(d_inv_sqrt_r)

        if matrix.shape[0] == matrix.shape[1]:
            return (d_r @ matrix).transpose() @ d_r

        degree_c = np.asarray(matrix.sum(axis=0)).flatten()
        d_inv_sqrt_c = np.power(degree_c, -0.5, where=degree_c > 0)
        d_inv_sqrt_c[~np.isfinite(d_inv_sqrt_c)] = 0.0
        return d_r @ matrix @ sp.diags(d_inv_sqrt_c)

    @staticmethod
    def _unique_undirected_pairs(row: np.ndarray, col: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
        low = np.minimum(row, col)
        high = np.maximum(row, col)
        hashes = low.astype(np.int64) * np.int64(n_nodes) + high.astype(np.int64)
        unique = np.unique(hashes)
        col_u = (unique % n_nodes).astype(np.int64)
        row_u = ((unique - col_u) // n_nodes).astype(np.int64)
        return row_u, col_u

    def _fewshot_file(self) -> Path | None:
        candidates = [
            self.root / f"partial_mat_{self.fewshot_ratio}.pkl",
            self.root / f"fewshot_mat_{self.fewshot_ratio}.pkl",
        ]
        if isinstance(self.fewshot_ratio, float):
            candidates.append(self.root / f"partial_mat_{self.fewshot_ratio:g}.pkl")
        for path in candidates:
            if path.exists():
                return path
        return None

    def _base_matrix(self) -> coo_matrix:
        if self.use_fewshot:
            fewshot = self._fewshot_file()
            if fewshot is not None:
                return self._load_coo(fewshot)

        matrix: coo_matrix | None = None
        for path in (self.train_file, self.val_file, self.test_file):
            matrix = self._load_coo(path) if matrix is None else matrix + self._load_coo(path)
        if matrix is None:
            raise FileNotFoundError(f"No split matrices found for {self.name} under {self.root}")
        return matrix.tocoo()

    def _features(self, num_nodes: int, original_shape: tuple[int, int]) -> torch.Tensor:
        if self.feature_mode == "original":
            feats = self._load_feats(self.feat_file)
            if feats is not None:
                x = torch.as_tensor(feats, dtype=torch.float32)
                if x.size(0) == num_nodes:
                    return x
                if original_shape[0] + original_shape[1] == num_nodes and x.size(0) == original_shape[0]:
                    pad = torch.zeros((original_shape[1], x.size(1)), dtype=x.dtype)
                    return torch.cat([x, pad], dim=0)
        if self.feature_mode not in {"constant", "original"}:
            raise ValueError(f"Unsupported feature_mode={self.feature_mode!r}")
        return torch.ones((num_nodes, self.constant_feature_dim), dtype=torch.float32)

    def to_data(self) -> "Data":
        from torch_geometric.data import Data

        matrix = self._base_matrix()
        original_shape = matrix.shape

        if matrix.shape[0] == matrix.shape[1]:
            row = np.concatenate([matrix.row, matrix.col]).astype(np.int64)
            col = np.concatenate([matrix.col, matrix.row]).astype(np.int64)
            row, col = self._unique_undirected_pairs(row, col, matrix.shape[0])
            structural = coo_matrix((np.ones_like(row, dtype=np.float32), (row, col)), shape=matrix.shape)
            if self.add_self_loops:
                structural = (structural + sp.eye(structural.shape[0], dtype=np.float32)).tocoo()
            norm = self._normalize_adj(structural).tocoo()
            num_nodes = structural.shape[0]
        else:
            num_src, num_dst = matrix.shape
            upper = sp.hstack([sp.csr_matrix((num_src, num_src), dtype=np.float32), matrix], format="csr")
            lower = sp.hstack([matrix.transpose(), sp.csr_matrix((num_dst, num_dst), dtype=np.float32)], format="csr")
            structural = (sp.vstack([upper, lower], format="csr") != 0).astype(np.float32).tocoo()
            if self.add_self_loops:
                structural = (structural + sp.eye(structural.shape[0], dtype=np.float32)).tocoo()
            norm = self._normalize_adj(structural).tocoo()
            num_nodes = num_src + num_dst

        edge_index = torch.as_tensor(np.vstack([norm.row, norm.col]), dtype=torch.long)
        data = Data(x=self._features(num_nodes, original_shape), edge_index=edge_index)
        data.edge_weight = torch.as_tensor(norm.data, dtype=torch.float32)
        data.dataset_name = self.name
        return data

    def splits(self) -> tuple["Data", "Data", "Data"]:
        from torch_geometric.transforms import RandomLinkSplit

        with temp_seed(self.split_seed):
            return RandomLinkSplit()(self.to_data())


class MultiDatasetLoader:
    """Round-robin neighbor-loader over multiple PyG link-prediction graphs."""

    def __init__(
        self,
        datasets: dict[str, "Data"] | list["Data"],
        batch_size: int = 8192,
        num_neighbors: list[int] | tuple[int, ...] = (32,),
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int | None = 8,
        persistent_workers: bool = True,
        test_mode: bool = False,
    ) -> None:
        from torch_geometric.loader import LinkNeighborLoader

        if isinstance(datasets, list):
            datasets = dict(enumerate(datasets))
        loader_kwargs = {
            "num_neighbors": list(num_neighbors),
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = persistent_workers
        self.loaders = {
            name: LinkNeighborLoader(
                data,
                edge_label=data.edge_label,
                edge_label_index=data.edge_label_index,
                **loader_kwargs,
            )
            for name, data in datasets.items()
        }
        if not self.loaders:
            raise ValueError("MultiDatasetLoader requires at least one dataset")
        self.len = max(len(loader) for loader in self.loaders.values())
        self.names = list(self.loaders.keys())
        self.test_mode = test_mode
        self._iters: dict[str, Iterator] = {}
        self._step = 0

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> "MultiDatasetLoader":
        self._iters = {name: iter(loader) for name, loader in self.loaders.items()}
        self._step = 0
        return self

    def __next__(self):
        from torch_geometric.data import Batch

        if self._step >= self.len:
            raise StopIteration

        batches = []
        per_dataset: dict[str, Any | None] = {}
        for name in self.names:
            try:
                batch = next(self._iters[name])
            except StopIteration:
                if self.test_mode:
                    batch = None
                else:
                    self._iters[name] = iter(self.loaders[name])
                    batch = next(self._iters[name])
            per_dataset[name] = batch
            if batch is not None:
                batch.dataset_name = name
                batches.append(batch)

        self._step += 1
        return Batch.from_data_list(batches), per_dataset


class FullGraphMultiDatasetLoader:
    """Edge-batch iterator that keeps each graph as a full-graph forward pass."""

    def __init__(
        self,
        datasets: dict[str, "Data"],
        edge_batch_size: int = 8192,
        shuffle: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        if not datasets:
            raise ValueError("FullGraphMultiDatasetLoader requires at least one dataset")
        self.datasets = datasets
        self.edge_batch_size = edge_batch_size
        self.shuffle = shuffle
        self.device = torch.device(device)
        self.names = list(datasets.keys())
        self.len = max(
            (data.edge_label.size(0) + edge_batch_size - 1) // edge_batch_size
            for data in datasets.values()
        )

    def __len__(self) -> int:
        return self.len

    def __iter__(self):
        order = {}
        for name, data in self.datasets.items():
            n_edges = data.edge_label.size(0)
            order[name] = torch.randperm(n_edges) if self.shuffle else torch.arange(n_edges)
        for step in range(self.len):
            items = []
            for name, data in self.datasets.items():
                start = step * self.edge_batch_size
                stop = min(start + self.edge_batch_size, data.edge_label.size(0))
                if start >= stop:
                    continue
                edge_idx = order[name][start:stop]
                items.append(
                    (
                        name,
                        data.to(self.device),
                        data.edge_label[edge_idx].to(self.device),
                        data.edge_label_index[:, edge_idx].to(self.device),
                    )
                )
            yield items


def load_link_prediction_splits(
    dataset_names: Iterable[str],
    *,
    base_dir: str | Path = DEFAULT_DATA_ROOT,
    split: str = "train",
    **dataset_kwargs,
) -> dict[str, "Data"]:
    split_idx = {"train": 0, "val": 1, "valid": 1, "test": 2}
    if split not in split_idx:
        raise ValueError(f"split must be one of {sorted(split_idx)}, got {split!r}")
    outputs = {}
    for name in dataset_names:
        outputs[name] = LinkPredictionDataset(name, base_dir=base_dir, **dataset_kwargs).splits()[split_idx[split]]
    return outputs

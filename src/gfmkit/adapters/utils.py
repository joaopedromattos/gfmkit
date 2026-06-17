from __future__ import annotations

import importlib
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def init_with_seed(seed: int, fn: Callable[[], T]) -> T:
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    py_state = random.getstate()
    seed_everything(seed)
    value = fn()
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    np.random.set_state(np_state)
    random.setstate(py_state)
    return value


def amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def split_pos_neg(edge_label: torch.Tensor, edge_label_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos = edge_label_index[:, edge_label == 1]
    neg = edge_label_index[:, edge_label == 0]
    return pos, neg


def dot_link_loss(z: torch.Tensor, edge_label: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
    pos_edge_index, neg_edge_index = split_pos_neg(edge_label, edge_label_index)
    pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
    return pos_loss + neg_loss


@contextmanager
def temporary_sys_path(*paths: str | Path) -> Iterator[None]:
    inserted = [str(Path(path).resolve()) for path in paths]
    old_path = list(sys.path)
    for path in reversed(inserted):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = old_path


def import_from_path(module_name: str, *paths: str | Path):
    with temporary_sys_path(*paths):
        return importlib.import_module(module_name)


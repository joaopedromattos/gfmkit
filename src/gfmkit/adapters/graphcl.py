from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool
from tqdm import tqdm

from gfmkit.data import MultiDatasetLoader
from gfmkit.outputs import EmbeddingOutput

from .base import GFMModel
from .utils import import_from_path, init_with_seed


def make_gin_conv(input_dim: int, out_dim: int):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(make_gin_conv(in_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        proj_dim = hidden_dim * num_layers
        self.project = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.ReLU(inplace=True), nn.Linear(proj_dim, proj_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(h, batch) for h in zs]
        z_cat, g_cat = [torch.cat(tensors, dim=1) for tensors in (zs, gs)]
        return z_cat, g_cat


class GraphCLEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, augmentor):
        super().__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        aug1, aug2 = self.augmentor
        x1, edge_index1, _ = aug1(x, edge_index)
        x2, edge_index2, _ = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


class GraphCLAdapter(GFMModel):
    """In-process GraphCL adapter using PyGCL when available."""

    def __init__(self, config):
        super().__init__(config)
        self._losses = None
        self._augmentors = None
        self._contrast_cls = None
        self.encoder_model = init_with_seed(self.seed, self._make_model).to(self.device)
        self.contrast_model = self._make_contrast().to(self.device)

    def _load_backend(self):
        if self._losses is not None:
            return
        try:
            self._losses = import_from_path("GCL.losses")
            self._augmentors = import_from_path("GCL.augmentors")
            models = import_from_path("GCL.models")
        except ModuleNotFoundError as exc:
            raise RuntimeError("GraphCLAdapter requires the optional PyGCL dependency.") from exc
        self._contrast_cls = models.DualBranchContrast

    def _make_model(self):
        self._load_backend()
        aug1 = self._augmentors.Identity()
        aug2 = self._augmentors.RandomChoice(
            [
                self._augmentors.RWSampling(num_seeds=1000, walk_length=10),
                self._augmentors.NodeDropping(pn=0.2),
                self._augmentors.FeatureMasking(pf=0.2),
                self._augmentors.EdgeRemoving(pe=0.2),
            ],
            1,
        )
        gconv = GConv(
            input_dim=self.hparam("in_channels", 128),
            hidden_dim=self.hparam("hidden_dim", self.hparam("hidden_channels", 32)),
            num_layers=self.hparam("num_layers", 2),
        )
        return GraphCLEncoder(gconv, (aug1, aug2))

    def _make_contrast(self):
        self._load_backend()
        return self._contrast_cls(loss=self._losses.InfoNCE(tau=self.hparam("tau", 0.2)), mode="G2G")

    def _prepare_features(self, data: Data) -> Data:
        if self.hparam("feature_mode", "random") == "random":
            generator = torch.Generator(device=self.device).manual_seed(self.seed * 1000 + data.num_nodes)
            data.x = torch.randn((data.num_nodes, self.hparam("in_channels", 128)), generator=generator, device=self.device)
        return data

    def pretrain(self, datasets: str | Iterable[str] = "link1"):
        self.ensure_dirs()
        epochs = int(self.hparam("pretrain_epochs", self.hparam("epochs", 100)))
        if epochs <= 0:
            return self
        train_data = self.load_splits(datasets, "train", feature_mode="constant")
        loader = MultiDatasetLoader(
            train_data,
            batch_size=self.hparam("batch_size", 8192),
            shuffle=True,
            num_workers=self.hparam("num_workers", 0),
        )
        optimizer = Adam(self.encoder_model.parameters(), lr=self.hparam("lr", 0.001))
        for epoch in tqdm(range(epochs), desc="GraphCL pretrain"):
            total = 0.0
            steps = 0
            self.encoder_model.train()
            for data, _per_ds in loader:
                data = self._prepare_features(data.to(self.device))
                optimizer.zero_grad(set_to_none=True)
                _z, _g, z1, z2, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch)
                g1, g2 = [self.encoder_model.encoder.project(g) for g in (g1, g2)]
                loss = self.contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, batch=data.batch)
                loss.backward()
                optimizer.step()
                total += float(loss.item())
                steps += 1
            if self.hparam("verbose", True):
                print(f"[GraphCL pretrain {epoch:03d}] loss={total / max(steps, 1):.4f}")
        self.save_checkpoint()
        return self

    @torch.no_grad()
    def embed(self, data: Data) -> torch.Tensor:
        self.encoder_model.eval()
        data = data.to(self.device)
        if getattr(data, "batch", None) is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
        data = self._prepare_features(data)
        z, _g, _z1, _z2, _g1, _g2 = self.encoder_model(data.x, data.edge_index, data.batch)
        return z.detach()

    def infer(self, datasets: str | Iterable[str] = "link2") -> dict[str, EmbeddingOutput]:
        outputs = {}
        for name in tqdm(self.resolve_datasets(datasets), desc="GraphCL infer"):
            data = self.load_split(name, "test", feature_mode="constant")
            z = self.embed(data)
            outputs[name] = EmbeddingOutput(
                z=z.detach().cpu(),
                edge_label=data.edge_label.detach().cpu(),
                edge_label_index=data.edge_label_index.detach().cpu(),
                dataset=name,
                model_name=self.name,
                seed=self.seed,
            )
        if self.hparam("save_outputs", True):
            self.save_outputs(outputs)
        return outputs

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path is not None else self.checkpoint_path(f"pretrain_{self.seed}.pt")
        torch.save(
            {
                "encoder_state_dict": self.encoder_model.state_dict(),
                "seed": self.seed,
                "hyperparams": dict(self.config.hyperparams),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(Path(path), map_location="cpu")
        self.encoder_model.load_state_dict(ckpt["encoder_state_dict"])
        self.encoder_model = self.encoder_model.to(self.device)
        return self


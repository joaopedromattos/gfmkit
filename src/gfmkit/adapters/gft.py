from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GAE
from tqdm import tqdm

from gfmkit.data import MultiDatasetLoader
from gfmkit.outputs import EmbeddingOutput

from .base import GFMModel
from .utils import amp_enabled, dot_link_loss, import_from_path, init_with_seed, split_pos_neg


class GFTAdapter(GFMModel):
    """In-process adapter for the local GFT implementation."""

    def __init__(self, config):
        super().__init__(config)
        self._encoder_cls = None
        self._vq_cls = None
        self.encoder = None
        self.vq = None
        self.model = None
        self.dataset_vqs: dict[str, torch.nn.Module] = {}
        self._build()

    def _load_backend(self):
        if self._encoder_cls is not None:
            return
        gft_root = self.config.baselines_root / "GFT" / "GFT"
        encoder_mod = import_from_path("model.encoder", gft_root)
        vq_mod = import_from_path("model.vq", gft_root)
        self._encoder_cls = encoder_mod.Encoder
        self._vq_cls = vq_mod.VectorQuantize

    def _make_encoder(self):
        self._load_backend()
        return self._encoder_cls(
            input_dim=self.hparam("in_channels", 128),
            hidden_dim=self.hparam("hidden_channels", 64),
            activation=nn.ReLU,
            num_layers=self.hparam("num_layers", 2),
            backbone=self.hparam("backbone", "gin"),
            normalize=self.hparam("normalize", "batch"),
            dropout=self.hparam("encoder_dropout", 0.0),
        )

    def _make_vq(self):
        self._load_backend()
        hidden_dim = self.hparam("hidden_channels", 64)
        return self._vq_cls(
            dim=hidden_dim,
            codebook_size=self.hparam("codebook_size", 128),
            codebook_dim=self.hparam("codebook_dim", hidden_dim),
            heads=self.hparam("codebook_heads", 1),
            separate_codebook_per_head=True,
            decay=0.8,
            commitment_weight=self.hparam("commitment_weight", 1.0),
            use_cosine_sim=True,
            learnable_codebook=True,
            orthogonal_reg_weight=self.hparam("ortho_reg_weight", 0.0),
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=self.hparam("ortho_reg_max_codes", None),
            kmeans_init=False,
            ema_update=False,
        )

    def _build(self):
        self.encoder = init_with_seed(self.seed, self._make_encoder)
        self.vq = init_with_seed(self.seed, self._make_vq)
        self.model = GAE(self.encoder).to(self.device)
        self.vq = self.vq.to(self.device)

    def pretrain(self, datasets: str | Iterable[str] = "link1"):
        self.ensure_dirs()
        epochs = int(self.hparam("pretrain_epochs", self.hparam("epochs", 100)))
        if epochs <= 0:
            return self
        train_data = self.load_splits(datasets, "train", feature_mode=self.hparam("feature_mode", "constant"))
        loader = MultiDatasetLoader(
            train_data,
            batch_size=self.hparam("batch_size", 8192),
            shuffle=True,
            num_workers=self.hparam("num_workers", 0),
        )
        optimizer = Adam(
            list(self.model.parameters()) + list(self.vq.parameters()),
            lr=self.hparam("pretrain_lr", self.hparam("lr", 0.001)),
            weight_decay=self.hparam("pretrain_wd", self.hparam("weight_decay", 5e-4)),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled(self.device))
        for epoch in tqdm(range(epochs), desc="GFT pretrain"):
            total, steps = 0.0, 0
            self.model.train()
            self.vq.train()
            for batch, _per_ds in loader:
                data = batch.to(self.device)
                with torch.cuda.amp.autocast(enabled=amp_enabled(self.device)):
                    z = self.model.encode(data.x, data.edge_index)
                    quantized, _indices, vq_loss, _orig = self.vq(z)
                    pos_edge_index, neg_edge_index = split_pos_neg(data.edge_label, data.edge_label_index)
                    recon_loss = self.model.recon_loss(quantized, pos_edge_index, neg_edge_index)
                    loss = recon_loss + self.hparam("vq_weight", 1.0) * vq_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                total += float(loss.item())
                steps += 1
            if self.hparam("verbose", True):
                print(f"[GFT pretrain {epoch:03d}] loss={total / max(steps, 1):.4f}")
        self.save_checkpoint()
        return self

    def finetune(self, datasets: str | Iterable[str] = "link2"):
        for param in self.encoder.parameters():
            param.requires_grad = False
        epochs = int(self.hparam("tune_epochs", 50))
        if epochs <= 0:
            return self
        for name in tqdm(self.resolve_datasets(datasets), desc="GFT finetune"):
            train_data = self.load_split(name, "train", feature_mode=self.hparam("feature_mode", "constant")).to(self.device)
            ds_vq = deepcopy(self.vq).to(self.device)
            optimizer = Adam(ds_vq.parameters(), lr=self.hparam("tune_lr", 0.001))
            for _epoch in range(epochs):
                ds_vq.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    z_raw = self.encoder(train_data.x, train_data.edge_index)
                quantized, _indices, vq_loss, _orig = ds_vq(z_raw)
                loss = dot_link_loss(quantized, train_data.edge_label, train_data.edge_label_index)
                loss = loss + self.hparam("vq_weight", 1.0) * vq_loss
                loss.backward()
                optimizer.step()
            self.dataset_vqs[name] = deepcopy(ds_vq).cpu()
            torch.save(
                {"vq_state_dict": ds_vq.state_dict(), "seed": self.seed},
                self.checkpoint_path(f"vq_{name}_{self.seed}.pt"),
            )
        return self

    @torch.no_grad()
    def embed(self, data: Data, vq: torch.nn.Module | None = None) -> torch.Tensor:
        self.encoder.eval()
        data = data.to(self.device)
        z = self.encoder(data.x, data.edge_index)
        active_vq = (vq or self.vq).to(self.device).eval()
        quantized, _indices, _loss, _orig = active_vq(z)
        return quantized.detach()

    def infer(self, datasets: str | Iterable[str] = "link2") -> dict[str, EmbeddingOutput]:
        outputs = {}
        for name in tqdm(self.resolve_datasets(datasets), desc="GFT infer"):
            data = self.load_split(name, "test", feature_mode=self.hparam("feature_mode", "constant"))
            z = self.embed(data, self.dataset_vqs.get(name))
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
                "encoder_state_dict": self.encoder.state_dict(),
                "vq_state_dict": self.vq.state_dict(),
                "seed": self.seed,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(Path(path), map_location="cpu")
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.vq.load_state_dict(ckpt["vq_state_dict"])
        self.model = GAE(self.encoder).to(self.device)
        self.vq = self.vq.to(self.device)
        return self


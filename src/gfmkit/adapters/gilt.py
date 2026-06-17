from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm

from gfmkit.data import MultiDatasetLoader
from gfmkit.outputs import EmbeddingOutput

from .base import GFMModel
from .utils import amp_enabled, import_from_path, init_with_seed


def select_batch_context(edge_label: torch.Tensor, edge_label_index: torch.Tensor, k: int = 32):
    pos_idx = (edge_label == 1).nonzero(as_tuple=True)[0]
    neg_idx = (edge_label == 0).nonzero(as_tuple=True)[0]
    k_pos = min(k, len(pos_idx))
    k_neg = min(k, len(neg_idx))
    ctx_idx = torch.cat(
        [
            pos_idx[torch.randperm(len(pos_idx), device=edge_label.device)[:k_pos]],
            neg_idx[torch.randperm(len(neg_idx), device=edge_label.device)[:k_neg]],
        ]
    )
    target_mask = torch.ones(len(edge_label), dtype=torch.bool, device=edge_label.device)
    target_mask[ctx_idx] = False
    return (
        edge_label_index[:, ctx_idx],
        edge_label[ctx_idx].long(),
        edge_label_index[:, target_mask],
        edge_label[target_mask].long(),
    )


class GILTAdapter(GFMModel):
    """In-process adapter for the local GILT implementation."""

    def __init__(self, config):
        super().__init__(config)
        self._gcn_cls = None
        self._pfn_cls = None
        self.encoder = init_with_seed(self.seed, self._make_encoder).to(self.device)
        self.predictor = init_with_seed(self.seed, self._make_predictor).to(self.device)

    def _load_backend(self):
        if self._gcn_cls is not None:
            return
        mod = import_from_path("src.model", self.config.baselines_root / "GILT")
        self._gcn_cls = mod.GCN
        self._pfn_cls = mod.PFNPredictorNodeCls

    def _make_encoder(self):
        self._load_backend()
        return self._gcn_cls(
            in_feats=self.hparam("in_channels", 128),
            h_feats=self.hparam("hidden_channels", 64),
            norm=True,
            relu=True,
            prop_step=self.hparam("prop_step", 2),
            dropout=self.hparam("encoder_dropout", 0.2),
            multilayer=False,
            use_gin=True,
            res=True,
            norm_affine=True,
        )

    def _make_predictor(self):
        self._load_backend()
        return self._pfn_cls(
            hidden_dim=self.hparam("hidden_channels", 64),
            nhead=self.hparam("pfn_nhead", 1),
            num_layers=self.hparam("pfn_layers", 2),
            mlp_layers=self.hparam("pfn_mlp_layers", 2),
            dropout=self.hparam("pfn_dropout", 0.2),
            norm=False,
            separate_att=False,
            sim=self.hparam("pfn_sim", "dot"),
            padding=self.hparam("pfn_padding", "zero"),
            norm_affine=True,
            normalize=False,
        )

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
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=self.hparam("pretrain_lr", self.hparam("lr", 0.001)),
            weight_decay=self.hparam("pretrain_wd", self.hparam("weight_decay", 5e-4)),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled(self.device))
        context_k = self.hparam("context_k", 32)
        for epoch in tqdm(range(epochs), desc="GILT pretrain"):
            total, steps = 0.0, 0
            self.encoder.train()
            self.predictor.train()
            for batch, _per_ds in loader:
                data = batch.to(self.device)
                n_pos = (data.edge_label == 1).sum().item()
                n_neg = (data.edge_label == 0).sum().item()
                if n_pos < 2 or n_neg < 2:
                    continue
                with torch.cuda.amp.autocast(enabled=amp_enabled(self.device)):
                    z = self.encoder(data.x, data.edge_index)
                    ctx_ei, ctx_labels, tgt_ei, tgt_labels = select_batch_context(
                        data.edge_label, data.edge_label_index, k=context_k
                    )
                    if tgt_ei.size(1) == 0:
                        continue
                    ctx_edge_emb = z[ctx_ei[0]] * z[ctx_ei[1]]
                    tgt_edge_emb = z[tgt_ei[0]] * z[tgt_ei[1]]
                    prototypes = torch.zeros(2, z.size(-1), device=self.device, dtype=z.dtype)
                    for cls in range(2):
                        mask = ctx_labels == cls
                        if mask.any():
                            prototypes[cls] = ctx_edge_emb[mask].mean(dim=0)
                    logits, _ = self.predictor(
                        data,
                        ctx_edge_emb,
                        tgt_edge_emb,
                        ctx_labels,
                        prototypes,
                        task_type="link_prediction",
                    )
                    loss = F.cross_entropy(logits, tgt_labels.long())
                scaler.scale(loss).backward()
                clip_grad = self.hparam("clip_grad", 1.0)
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.predictor.parameters()),
                        clip_grad,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                total += float(loss.item())
                steps += 1
            if self.hparam("verbose", True):
                print(f"[GILT pretrain {epoch:03d}] loss={total / max(steps, 1):.4f}")
        self.save_checkpoint()
        return self

    @torch.no_grad()
    def embed(self, data: Data) -> torch.Tensor:
        self.encoder.eval()
        data = data.to(self.device)
        return self.encoder(data.x, data.edge_index).detach()

    def infer(self, datasets: str | Iterable[str] = "link2") -> dict[str, EmbeddingOutput]:
        outputs = {}
        for name in tqdm(self.resolve_datasets(datasets), desc="GILT infer"):
            data = self.load_split(name, "test", feature_mode=self.hparam("feature_mode", "constant"))
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
                "encoder_state_dict": self.encoder.state_dict(),
                "predictor_state_dict": self.predictor.state_dict(),
                "seed": self.seed,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(Path(path), map_location="cpu")
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        if "predictor_state_dict" in ckpt:
            self.predictor.load_state_dict(ckpt["predictor_state_dict"])
        return self


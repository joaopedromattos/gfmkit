from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GAE
from tqdm import tqdm

from gfmkit.data import FullGraphMultiDatasetLoader, MultiDatasetLoader
from gfmkit.outputs import EmbeddingOutput

from .base import GFMModel
from .pyg_blocks import (
    FeatureWeightedPrompt,
    GINEncoder,
    GPFPlusPrompt,
    LinearPrompt,
    PromptedGINEncoder,
    SimplePrompt,
)
from .utils import amp_enabled, dot_link_loss, init_with_seed, split_pos_neg


class GNNAdapter(GFMModel):
    """Plain GIN-GAE link-prediction encoder."""

    def __init__(self, config):
        super().__init__(config)
        self.encoder = init_with_seed(
            self.seed,
            lambda: GINEncoder(
                self.hparam("in_channels", 128),
                self.hparam("hidden_channels", 128),
                self.hparam("out_channels", 64),
            ),
        )
        self.model = GAE(self.encoder).to(self.device)

    def _train_loader(self, datasets: Iterable[str]):
        train_datasets = self.load_splits(
            datasets,
            split="train",
            feature_mode=self.hparam("feature_mode", "constant"),
        )
        if self.hparam("fullgraph", False):
            return FullGraphMultiDatasetLoader(
                train_datasets,
                edge_batch_size=self.hparam("batch_size", 8192),
                shuffle=True,
                device=self.device,
            )
        return MultiDatasetLoader(
            train_datasets,
            batch_size=self.hparam("batch_size", 8192),
            shuffle=True,
            num_workers=self.hparam("num_workers", 0),
        )

    def pretrain(self, datasets: str | Iterable[str] = "link1"):
        self.ensure_dirs()
        epochs = int(self.hparam("pretrain_epochs", self.hparam("epochs", 100)))
        if epochs <= 0:
            return self
        loader = self._train_loader(self.resolve_datasets(datasets))
        optimizer = Adam(
            self.model.parameters(),
            lr=self.hparam("pretrain_lr", self.hparam("lr", 0.001)),
            weight_decay=self.hparam("pretrain_wd", self.hparam("weight_decay", 5e-4)),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled(self.device))
        self.model.train()
        for epoch in tqdm(range(epochs), desc=f"{self.name} pretrain"):
            total_loss, steps = 0.0, 0
            for batch in loader:
                if isinstance(batch, list):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.zeros((), device=self.device)
                    for _name, data, edge_label, edge_label_index in batch:
                        with torch.cuda.amp.autocast(enabled=amp_enabled(self.device)):
                            z = self.model.encode(data.x, data.edge_index)
                            pos_edge_index, neg_edge_index = split_pos_neg(edge_label, edge_label_index)
                            loss = loss + self.model.recon_loss(z, pos_edge_index, neg_edge_index) / max(len(batch), 1)
                else:
                    data = batch[0].to(self.device) if isinstance(batch, tuple) else batch.to(self.device)
                    with torch.cuda.amp.autocast(enabled=amp_enabled(self.device)):
                        z = self.model.encode(data.x, data.edge_index)
                        pos_edge_index, neg_edge_index = split_pos_neg(data.edge_label, data.edge_label_index)
                        loss = self.model.recon_loss(z, pos_edge_index, neg_edge_index)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.item())
                steps += 1
            if (epoch + 1) % int(self.hparam("checkpoint_interval", 20)) == 0:
                self.save_checkpoint()
            if self.hparam("verbose", True):
                print(f"[{self.name} pretrain {epoch:03d}] loss={total_loss / max(steps, 1):.4f}")
        self.save_checkpoint()
        return self

    @torch.no_grad()
    def embed(self, data: Data) -> torch.Tensor:
        self.model.eval()
        data = data.to(self.device)
        return self.model.encode(data.x, data.edge_index).detach()

    def infer(self, datasets: str | Iterable[str] = "link2") -> dict[str, EmbeddingOutput]:
        outputs = {}
        for name in tqdm(self.resolve_datasets(datasets), desc=f"{self.name} infer"):
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
        torch.save({"encoder_state_dict": self.encoder.state_dict(), "seed": self.seed}, path)
        return path

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(Path(path), map_location="cpu")
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.model = GAE(self.encoder).to(self.device)
        return self


class _PromptAdapter(GNNAdapter):
    prompt_mode = "output"

    def __init__(self, config):
        GFMModel.__init__(self, config)
        encoder_cls = PromptedGINEncoder if self.prompt_mode == "input" else GINEncoder
        self.encoder = init_with_seed(
            self.seed,
            lambda: encoder_cls(
                self.hparam("in_channels", 128),
                self.hparam("hidden_channels", 128),
                self.hparam("out_channels", 64),
            ),
        )
        self.model = GAE(self.encoder).to(self.device)
        self.dataset_prompts: dict[str, torch.nn.Module] = {}

    def make_prompt(self) -> torch.nn.Module:
        raise NotImplementedError

    def _prompted_z(self, data: Data, prompt: torch.nn.Module) -> torch.Tensor:
        if self.prompt_mode == "input":
            return self.encoder(data.x, data.edge_index, prompt=prompt)
        with torch.no_grad():
            z_raw = self.encoder(data.x, data.edge_index)
        return prompt(z_raw)

    def finetune(self, datasets: str | Iterable[str] = "link2"):
        for param in self.encoder.parameters():
            param.requires_grad = False
        epochs = int(self.hparam("prompt_epochs", self.hparam("tune_epochs", 50)))
        if epochs <= 0:
            return self
        for name in tqdm(self.resolve_datasets(datasets), desc=f"{self.name} finetune"):
            train_data = self.load_split(name, "train", feature_mode=self.hparam("feature_mode", "constant")).to(self.device)
            prompt = self.make_prompt().to(self.device)
            optimizer = Adam(prompt.parameters(), lr=self.hparam("prompt_lr", self.hparam("tune_lr", 0.001)))
            for _epoch in range(epochs):
                prompt.train()
                optimizer.zero_grad(set_to_none=True)
                z = self._prompted_z(train_data, prompt)
                loss = dot_link_loss(z, train_data.edge_label, train_data.edge_label_index)
                loss.backward()
                optimizer.step()
            self.dataset_prompts[name] = deepcopy(prompt).cpu()
            torch.save(
                {"prompt_state_dict": prompt.state_dict(), "seed": self.seed, "model_name": self.name},
                self.checkpoint_path(f"prompt_{name}_{self.seed}.pt"),
            )
        return self

    @torch.no_grad()
    def embed(self, data: Data, prompt: torch.nn.Module | None = None) -> torch.Tensor:
        self.encoder.eval()
        data = data.to(self.device)
        if prompt is None:
            if self.prompt_mode == "input":
                return self.encoder(data.x, data.edge_index).detach()
            return self.encoder(data.x, data.edge_index).detach()
        prompt = prompt.to(self.device).eval()
        return self._prompted_z(data, prompt).detach()

    def infer(self, datasets: str | Iterable[str] = "link2") -> dict[str, EmbeddingOutput]:
        outputs = {}
        for name in tqdm(self.resolve_datasets(datasets), desc=f"{self.name} infer"):
            data = self.load_split(name, "test", feature_mode=self.hparam("feature_mode", "constant"))
            prompt = self.dataset_prompts.get(name)
            z = self.embed(data, prompt=prompt)
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


class GPFAdapter(_PromptAdapter):
    """GPF / GPF-plus input-prompt adapter."""

    prompt_mode = "input"

    def make_prompt(self) -> torch.nn.Module:
        dim = self.hparam("in_channels", 128)
        if self.hparam("tuning_type", "gpf") in {"gpf-plus", "gpf_plus", "plus"}:
            return GPFPlusPrompt(dim, pnum=self.hparam("pnum", 5))
        return SimplePrompt(dim)


class GraphPromptAdapter(_PromptAdapter):
    """GraphPrompt output-prompt adapter."""

    prompt_mode = "output"

    def make_prompt(self) -> torch.nn.Module:
        dim = self.hparam("out_channels", 64)
        if self.hparam("prompt_type", "feature-weighted") == "linear":
            return LinearPrompt(dim, dim)
        return FeatureWeightedPrompt(dim)

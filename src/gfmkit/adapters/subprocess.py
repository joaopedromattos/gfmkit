from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Iterable

import torch

from gfmkit.outputs import load_embedding_output

from .base import GFMModel


class SubprocessAdapter(GFMModel):
    """Adapter for baselines whose stable interface is still their original script."""

    supports_in_process = False
    baseline_dir_name: str = ""

    def baseline_dir(self) -> Path:
        return self.config.baselines_root / (self.baseline_dir_name or self.name)

    def default_command(self, phase: str, datasets: list[str]) -> list[str]:
        raise NotImplementedError

    def command(self, phase: str, datasets: str | Iterable[str]) -> list[str]:
        dataset_list = self.resolve_datasets(datasets)
        template = self.hparam(f"{phase}_command", self.hparam("command", None))
        if template:
            values = {
                "phase": phase,
                "seed": self.seed,
                "device": self.config.device,
                "gpu": str(self.config.device).split(":")[-1] if "cuda" in str(self.config.device) else "0",
                "datasets": ",".join(dataset_list),
                "data_root": str(self.config.data_root),
                "output_root": str(self.config.output_root),
                "checkpoint_root": str(self.config.checkpoint_root),
                "baseline_dir": str(self.baseline_dir()),
            }
            return shlex.split(str(template).format(**values))
        return self.default_command(phase, dataset_list)

    def _run(self, phase: str, datasets: str | Iterable[str]):
        cmd = self.command(phase, datasets)
        if self.hparam("dry_run", False):
            return subprocess.CompletedProcess(cmd, 0)
        self.config.model_log_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.run(
            cmd,
            cwd=self.baseline_dir(),
            check=True,
            text=True,
        )

    def pretrain(self, datasets: str | Iterable[str] = "link1"):
        return self._run("pretrain", datasets)

    def finetune(self, datasets: str | Iterable[str] = "link2"):
        command = self.hparam("finetune_command", None)
        if command is None and self.hparam("combined_train_infer", True):
            return None
        return self._run("finetune", datasets)

    def infer(self, datasets: str | Iterable[str] = "link2") -> dict[str, dict]:
        if self.hparam("run_infer_command", False):
            self._run("infer", datasets)
        outputs = {}
        strict = self.hparam("strict_outputs", True)
        for name in self.resolve_datasets(datasets):
            candidates = [
                self.config.output_root / self.name / f"{name}_{self.seed}.pt",
                self.config.output_root / self.name / f"{name}.pt",
                self.baseline_dir() / "outputs" / f"{name}_{self.seed}.pt",
                self.baseline_dir() / "outputs" / f"{name}.pt",
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                if strict:
                    checked = ", ".join(str(candidate) for candidate in candidates)
                    raise FileNotFoundError(f"{self.name} did not produce an output for {name}. Checked: {checked}")
                continue
            outputs[name] = load_embedding_output(path)
        return outputs


class AnyGraphAdapter(SubprocessAdapter):
    baseline_dir_name = "AnyGraph"

    def default_command(self, phase: str, datasets: list[str]) -> list[str]:
        script = "evaluate.py" if phase == "infer" else "main.py"
        gpu = str(self.config.device).split(":")[-1] if "cuda" in str(self.config.device) else "0"
        return [
            "python",
            script,
            "--seed",
            str(self.seed),
            "--gpu",
            gpu,
            "--epoch",
            str(self.hparam("epochs", self.hparam("pretrain_epochs", 100))),
        ]


class UniLPAdapter(SubprocessAdapter):
    baseline_dir_name = "UniLP"

    def default_command(self, phase: str, datasets: list[str]) -> list[str]:
        suite_pretrain = ",".join(self.resolve_datasets(self.hparam("pretrain_datasets", "link1")))
        suite_infer = ",".join(datasets)
        return [
            "python",
            "main.py",
            "--pretrain_datasets",
            suite_pretrain,
            "--inference_datasets",
            suite_infer,
            "--dataset_dir",
            str(self.config.data_root),
            "--epochs",
            str(self.hparam("epochs", self.hparam("pretrain_epochs", 50))),
            "--runs",
            str(self.hparam("runs", 1)),
            "--save_model",
            str(self.config.model_checkpoint_dir),
        ]


class GraphAnyAdapter(SubprocessAdapter):
    baseline_dir_name = "GraphAny"

    def default_command(self, phase: str, datasets: list[str]) -> list[str]:
        # GraphAny is a node-classification GFM. The adapter exposes it through
        # the same registry, but users should pass Hydra overrides when they want
        # a real GraphAny run for a specific dataset lookup.
        overrides = list(self.hparam("hydra_overrides", []))
        if not overrides:
            overrides = [
                f"seed={self.seed}",
                f"total_steps={self.hparam('total_steps', 0)}",
            ]
        return ["python", "graphany/run.py", *overrides]


class MochiAdapter(SubprocessAdapter):
    """Adapter for the root-level Mochi / Mochi++ implementation."""

    baseline_dir_name = "Mochi"

    def baseline_dir(self) -> Path:
        return self.config.repo_root / "Mochi"

    def _gpu_id(self) -> str:
        return str(self.config.device).split(":")[-1] if "cuda" in str(self.config.device) else "0"

    def _dataset_setting(self, phase: str, datasets: list[str]) -> str:
        if self.hparam(f"{phase}_dataset_setting", None):
            return self.hparam(f"{phase}_dataset_setting")
        if datasets == self.resolve_datasets("link1"):
            return "link1"
        if datasets == self.resolve_datasets("link2"):
            return "link2"
        if datasets == self.resolve_datasets("smoke"):
            return "smoke"
        if len(datasets) == 1:
            return datasets[0]
        return ",".join(datasets)

    def default_command(self, phase: str, datasets: list[str]) -> list[str]:
        cmd = [
            "python",
            "train.py",
            "--model_variant",
            self.hparam("model_variant", "mochi++"),
            "--seed",
            str(self.seed),
            "--gpu",
            self._gpu_id(),
            "--dataset_setting",
            self._dataset_setting(phase, datasets),
            "--data_root",
            str(self.hparam("mochi_data_root", self.config.repo_root / "data")),
            "--lp_data_root",
            str(self.hparam("lp_data_root", self.config.data_root)),
        ]

        if self.hparam("cache_dir", None):
            cmd.extend(["--cache_dir", str(self.hparam("cache_dir"))])
        if self.hparam("cstag_root", None):
            cmd.extend(["--cstag_root", str(self.hparam("cstag_root"))])
        if self.hparam("nc_datasets", None) is not None:
            cmd.append("--nc_datasets")
            cmd.extend([str(x) for x in self.hparam("nc_datasets")])
        if self.hparam("gc_datasets", None) is not None:
            cmd.append("--gc_datasets")
            cmd.extend([str(x) for x in self.hparam("gc_datasets")])
        if phase in {"infer", "finetune"} or self.hparam("eval_only", False):
            cmd.append("--eval_only")
            load_model = self.hparam("load_model", None)
            if load_model:
                cmd.extend(["--load_model", str(load_model)])
            elif self.hparam("load_pretrained", False):
                cmd.append("--load_pretrained")
        else:
            cmd.extend(["--train_steps", str(self.hparam("train_steps", self.hparam("pretrain_steps", 12991)))])
        if self.hparam("no_save_embeddings", False):
            cmd.append("--no_save_embeddings")
        return cmd

    def pretrain(self, datasets: str | Iterable[str] = "link1"):
        # Mochi's train.py performs train + eval + embedding export in one run.
        return self._run("pretrain", datasets)

    def finetune(self, datasets: str | Iterable[str] = "link2"):
        # Mochi does not have a separate fine-tuning phase in this codebase.
        if self.hparam("run_finetune_command", False):
            return self._run("finetune", datasets)
        return None

    def infer(self, datasets: str | Iterable[str] = "link2") -> dict[str, dict]:
        if self.hparam("run_infer_command", False):
            self._run("infer", datasets)
        outputs = {}
        strict = self.hparam("strict_outputs", True)
        for name in self.resolve_datasets(datasets):
            candidates = [
                self.baseline_dir() / "outputs" / f"lp_{name}_{self.seed}.pt",
                self.config.output_root / self.name / f"lp_{name}_{self.seed}.pt",
                self.config.output_root / self.name / f"{name}_{self.seed}.pt",
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                if strict:
                    checked = ", ".join(str(candidate) for candidate in candidates)
                    raise FileNotFoundError(f"Mochi did not produce an LP embedding for {name}. Checked: {checked}")
                continue
            z = torch.load(path, map_location="cpu")
            outputs[name] = {
                "z": z,
                "dataset": name,
                "model_name": self.name,
                "seed": self.seed,
                "source_path": str(path),
            }
        return outputs

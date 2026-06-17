from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Type

from .adapters.base import GFMModel
from .config import ExperimentConfig

AdapterRef = Type[GFMModel] | str


@dataclass
class GFMRegistry:
    _models: dict[str, AdapterRef] = field(default_factory=dict)
    _aliases: dict[str, str] = field(default_factory=dict)

    def register(self, name: str, adapter: AdapterRef, *aliases: str) -> None:
        canonical = name
        self._models[canonical] = adapter
        for key in (name, *aliases):
            self._aliases[self._normalize(key)] = canonical

    @staticmethod
    def _normalize(name: str) -> str:
        return name.replace("_", "").replace("-", "").lower()

    def canonical_name(self, name: str) -> str:
        try:
            return self._aliases[self._normalize(name)]
        except KeyError as exc:
            raise KeyError(f"Unknown GFM model {name!r}. Available: {', '.join(self.list_models())}") from exc

    @staticmethod
    def _resolve_adapter(adapter: AdapterRef) -> Type[GFMModel]:
        if not isinstance(adapter, str):
            return adapter
        module_name, class_name = adapter.rsplit(":", 1)
        return getattr(import_module(module_name), class_name)

    def create(self, name: str, config: ExperimentConfig) -> GFMModel:
        canonical = self.canonical_name(name)
        config.model_name = canonical
        return self._resolve_adapter(self._models[canonical])(config)

    def list_models(self) -> list[str]:
        return sorted(self._models)


def get_default_registry() -> GFMRegistry:
    registry = GFMRegistry()
    registry.register("GNN", "gfmkit.adapters.standard:GNNAdapter", "GNNs", "GIN", "GIN-GAE")
    registry.register("GraphCL", "gfmkit.adapters.graphcl:GraphCLAdapter")
    registry.register("GPF", "gfmkit.adapters.standard:GPFAdapter", "GPFPlus", "GPF-Plus")
    registry.register("GraphPrompt", "gfmkit.adapters.standard:GraphPromptAdapter")
    registry.register("GFT", "gfmkit.adapters.gft:GFTAdapter")
    registry.register("GILT", "gfmkit.adapters.gilt:GILTAdapter")
    registry.register("AnyGraph", "gfmkit.adapters.subprocess:AnyGraphAdapter")
    registry.register("GraphAny", "gfmkit.adapters.subprocess:GraphAnyAdapter")
    registry.register("Mochi", "gfmkit.adapters.subprocess:MochiAdapter", "Mochi++", "MochiPlus")
    registry.register("UniLP", "gfmkit.adapters.subprocess:UniLPAdapter")
    return registry

from .base import GFMModel

__all__ = [
    "AnyGraphAdapter",
    "GFMModel",
    "GFTAdapter",
    "GILTAdapter",
    "GNNAdapter",
    "GPFAdapter",
    "GraphAnyAdapter",
    "GraphCLAdapter",
    "GraphPromptAdapter",
    "MochiAdapter",
    "SubprocessAdapter",
    "UniLPAdapter",
]


def __getattr__(name: str):
    mapping = {
        "AnyGraphAdapter": "gfmkit.adapters.subprocess",
        "GFTAdapter": "gfmkit.adapters.gft",
        "GILTAdapter": "gfmkit.adapters.gilt",
        "GNNAdapter": "gfmkit.adapters.standard",
        "GPFAdapter": "gfmkit.adapters.standard",
        "GraphAnyAdapter": "gfmkit.adapters.subprocess",
        "GraphCLAdapter": "gfmkit.adapters.graphcl",
        "GraphPromptAdapter": "gfmkit.adapters.standard",
        "MochiAdapter": "gfmkit.adapters.subprocess",
        "SubprocessAdapter": "gfmkit.adapters.subprocess",
        "UniLPAdapter": "gfmkit.adapters.subprocess",
    }
    if name not in mapping:
        raise AttributeError(name)
    from importlib import import_module

    return getattr(import_module(mapping[name]), name)

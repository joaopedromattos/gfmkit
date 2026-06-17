from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root, preferring the first parent with AGENTS.md."""
    here = (start or Path.cwd()).resolve()
    for path in (here, *here.parents):
        if (path / "AGENTS.md").exists() and (path / "baselines").exists():
            return path
    return here


REPO_ROOT = find_repo_root()
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "zero-shot datasets"
DEFAULT_BASELINES_ROOT = REPO_ROOT / "baselines"
DEFAULT_OUTPUT_ROOT = DEFAULT_BASELINES_ROOT / "outputs"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "gfmkit"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "gfmkit"


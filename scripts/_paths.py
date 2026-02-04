from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def runs_root() -> Path:
    env = os.getenv("RW_RUNS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "runs"

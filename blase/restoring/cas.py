from __future__ import annotations
from pathlib import Path

def cas_root(run_path: Path) -> Path:
    return run_path / "cas" / "sha256"

def path_for(run_path: Path, kind: str, data_hash: str) -> Path:
    root = cas_root(run_path) / kind
    return root / data_hash[:2] / data_hash[2:]
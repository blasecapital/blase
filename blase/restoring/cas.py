from __future__ import annotations
from pathlib import Path
import os
import tempfile
import shutil


def cas_root(run_path: Path) -> Path:
    return run_path / "cas" / "sha256"


def path_for(run_path: Path, kind: str, data_hash: str) -> Path:
    root = cas_root(run_path) / kind
    return root / data_hash[:2] / data_hash[2:]


def write_bytes(
    run_path: Path,
    kind: str,
    data_hash: str,
    data: bytes,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Materialize `data` into CAS at the location implied by `data_hash`.

    Writes atomically via a temp file + rename.
    If the target exists and `overwrite` is False, it’s left untouched.
    """
    dest = path_for(run_path, kind, data_hash)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        return dest

    # atomic write
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(dest.parent)) as tf:
        tmp_path = Path(tf.name)
        tf.write(data)
        tf.flush()
        os.fsync(tf.fileno())
    try:
        # On some filesystems replace may not be atomic; fall back to move.
        try:
            os.replace(tmp_path, dest)
        except Exception:
            shutil.move(str(tmp_path), str(dest))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return dest

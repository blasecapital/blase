from __future__ import annotations
from typing import Optional, List
from pathlib import Path
import os, shutil

from blase.utils.hashing import Hash
from blase.utils import config
from blase.restoring import store, cas
from .io_safety import resolve_conflict_path

class NeedReplay(FileNotFoundError):
    """Raised when a data hash has no valid local copy; caller should replay."""

def _valid_local_candidates(run_path: Path, data_hash: str) -> List[Path]:
    hasher = Hash()
    out: List[Path] = []

    # 1) known materializations
    for p in store.load_materializations(run_path, data_hash):
        pp = Path(p)
        if pp.exists() and hasher.hash_file(pp) == data_hash:
            out.append(pp)
        elif pp.exists():  # hash mismatch; prune stale record (optional)
            try:
                store.drop_materialization(run_path, data_hash, p)
            except Exception:
                pass

    # 2) original source_path (if any)
    src = store.load_data_source_path(run_path, data_hash)
    if src:
        ps = Path(src)
        if ps.exists() and hasher.hash_file(ps) == data_hash:
            out.append(ps)

    return out

def ensure_local(run_path: Path, data_hash: str, *, kind: str = "data",
                 policy: str = "reuse", to_dir: Optional[Path] = None,
                 target_name: Optional[str] = None,
                 on_conflict: Optional[str] = None) -> Path:
    hasher = Hash()

    # Code/env blobs still come from CAS
    if kind in ("code", "env"):
        cas_path = cas.path_for(run_path, kind=kind, data_hash=data_hash)
        if not cas_path.exists():
            raise FileNotFoundError(f"missing {kind} blob in CAS: {cas_path}")
        return cas_path  # usually we just return the CAS path for callables/env

    # Data kinds: prefer local materializations/source-of-truth
    candidates = _valid_local_candidates(run_path, data_hash)
    if not candidates:
        # Nothing local; caller should replay
        raise NeedReplay(f"no local materialization for data {data_hash}")

    src = candidates[0]  # first valid path

    if to_dir is None:
        to_dir = config.RESTORE_DEFAULT_DIR

    # If caller passed a full filename (args.to), use that name. Otherwise use the src basename.
    name = target_name or os.path.basename(str(src))
    out = (to_dir / name).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Conflict policy (rename/overwrite/fail)
    out = resolve_conflict_path(out,
                                policy=(on_conflict or config.RESTORE_CONFLICT),
                                suffix=config.RESTORE_SUFFIX)

    # Try hardlink, else copy
    try:
        os.link(src, out)
    except OSError:
        shutil.copy2(src, out)

    if hasher.hash_file(out) != data_hash:
        raise IOError("materialized file hash mismatch")

    # Optionally (re)index this materialization
    try:
        store.record_materialization(run_path, data_hash, str(out))
    except Exception:
        pass

    return out
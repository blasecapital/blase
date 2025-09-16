from __future__ import annotations
from typing import Optional, List
from pathlib import Path
import os
import shutil

from blase.utils.hashing import Hash
from blase.utils import config
from blase.restoring import store, cas
from .io_safety import resolve_conflict_path


class NeedReplay(FileNotFoundError):
    """Raised when a data hash has no valid local copy; caller should replay."""


META_KINDS = (
    "image.manifest",
    "image.batch.meta",
    "dataset.checkpoint.meta",
    # add any other descriptor kinds you store as bytes in CAS
)


def _valid_local_candidates(run_path: Path, data_hash: str) -> List[Path]:
    """
    Return valid local file paths that match a given data hash.

    This helper checks known materializations and original source paths
    associated with a run. Only paths that exist and hash correctly are
    returned. Stale records are pruned if hash mismatches are detected.

    Parameters
    ----------
    run_path : Path
        Root path of the run containing metadata and materializations.
    data_hash : str
        Expected hash identifier of the data artifact.

    Returns
    -------
    List[Path]
        List of valid local file paths that match the hash.
    """
    hasher = Hash()
    out: List[Path] = []

    # 1) known materializations
    for p in store.load_materializations(run_path, data_hash):
        pp = Path(p)
        if pp.exists() and hasher.hash_file(pp) == data_hash:
            out.append(pp)
        elif pp.exists():
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


def ensure_local(
    run_path: Path,
    data_hash: str,
    *,
    kind: str = "data",
    policy: str = "reuse",
    to_dir: Optional[Path] = None,
    target_name: Optional[str] = None,
    on_conflict: Optional[str] = None,
) -> Path:
    """
    Ensure a data, code, or environment artifact is locally materialized.

    This function resolves a local copy of an artifact (by hash) from a run.
    For data artifacts, it prefers existing materializations or source-of-truth
    files; for code and environment blobs, it retrieves the content-addressable
    storage (CAS) path. If no valid local copy is available, the caller is
    expected to trigger a replay to regenerate the artifact.

    Parameters
    ----------
    run_path : Path
        Root path of the run containing CAS, metadata, and materialization logs.
    data_hash : str
        Expected hash identifier of the artifact to restore.
    kind : {"data", "code", "env"}, default="data"
        Type of artifact to materialize. Code and environment blobs always
        resolve to CAS paths, while data is restored from materializations.
    policy : str, default="reuse"
        Currently unused placeholder for higher-level reuse policies.
    to_dir : Path, optional
        Directory where the artifact should be materialized. Defaults to
        ``config.RESTORE_DEFAULT_DIR`` if not provided.
    target_name : str, optional
        Filename to use for the restored artifact. If not given, the basename
        of the source file is reused.
    on_conflict : {"rename", "overwrite", "fail"}, optional
        File conflict policy for the target. Falls back to
        ``config.RESTORE_CONFLICT`` if not provided.

    Returns
    -------
    Path
        Path to the ensured local artifact (hardlinked or copied into place).

    Raises
    ------
    FileNotFoundError
        If the requested code or environment blob is missing from CAS.
    NeedReplay
        If no valid local data materialization exists for the hash.
    IOError
        If the materialized file’s hash does not match the expected hash.

    Notes
    -----
    - Data artifacts are attempted first via known materializations or
      recorded source paths.
    - On conflict, the output path is resolved via
      :func:`resolve_conflict_path`.
    - The function prefers hardlinking to avoid duplication, falling back
      to a full copy if hardlinking fails.
    - Successfully materialized files are re-recorded into the run’s
      materialization index for future reuse.

    Examples
    --------
    >>> from pathlib import Path
    >>> ensure_local(Path("runs/2024-08-01T12-00-00"), "abc123hash")
    PosixPath('restore/output/data.csv')

    If no valid data exists locally:
    >>> ensure_local(Path("runs/2024-08-01T12-00-00"), "missinghash")
    Traceback (most recent call last):
        ...
    NeedReplay: no local materialization for data missinghash
    """
    hasher = Hash()

    if kind in ("code", "env") or kind in META_KINDS:
        p = cas.path_for(run_path, kind=kind, data_hash=data_hash)
        if not p.exists():
            raise FileNotFoundError(f"missing {kind} blob in CAS: {p}")
        return p

    # Data kinds: prefer local materializations/source-of-truth
    candidates = _valid_local_candidates(run_path, data_hash)
    if not candidates:
        raise NeedReplay(f"no local materialization for data {data_hash}")

    src = candidates[0]

    if to_dir is None:
        to_dir = config.RESTORE_DEFAULT_DIR

    # If caller passed a full filename (args.to), use that name. Otherwise use the src basename.
    name = target_name or os.path.basename(str(src))
    out = (to_dir / name).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Conflict policy (rename/overwrite/fail)
    out = resolve_conflict_path(
        out,
        policy=(on_conflict or config.RESTORE_CONFLICT),
        suffix=config.RESTORE_SUFFIX,
    )

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

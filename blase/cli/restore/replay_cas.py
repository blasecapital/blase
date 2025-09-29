from pathlib import Path
from typing import Optional, Set, List, Tuple
import tempfile

from blase.restoring import store, materialize
from blase.utils.hashing import Hash


def _assert_path_matches_hash(path: Path, expected_hash: str, what: str) -> None:
    got = Hash().hash_file(path)
    if got != expected_hash:
        raise SystemExit(
            f"restore: {what} hash mismatch: got {got}, expected {expected_hash}"
        )


def _resolve_source_no_copy(
    run_path: Path,
    source_hash: str,
    *,
    seen_steps: Optional[Set[str]] = None,
    created_paths: Optional[List[Path]] = None,
) -> Tuple[Path, bool]:
    """
    Return a path to the recorded source without copying when possible.
    If the original source_path exists and matches the recorded hash, use it.
    Else replay/materialize (this will copy into /restored); we return that path and mark created=True.
    """
    # Try original source_of_truth path first (no copy)
    src = store.load_data_source_path(run_path, source_hash)
    if src:
        p = Path(src)
        if p.exists() and Hash().hash_file(p) == source_hash:
            return p, False  # not created

    # Fallback: ensure locally (may trigger replay of the producer; will copy/link into /restored)
    from .replay import _ensure_data_local_or_replay

    p, created = _ensure_data_local_or_replay(
        run_path,
        source_hash,
        store.get_data_kind(run_path, source_hash) or "csv",
        seen_steps=seen_steps,
        created_paths=created_paths,
    )
    # Track for cleanup if we had to copy
    if created and created_paths is not None:
        created_paths.append(p)
    return p, created


def _resolve_seed_no_copy_or_ephemeral(
    run_path: Path,
    seed_hash: str,
    *,
    seen_steps: Optional[Set[str]] = None,
    created_paths: Optional[List[Path]] = None,
) -> Tuple[Path, bool]:
    """
    Return a path to the recorded seed (run-1 output) without persisting it in /restored.
    - If a valid existing path already exists (materialization or original), use it (no copy).
    - Otherwise, replay the seed's producer into a *temporary* file (not /restored) and
      do NOT record a materialization. Mark created=True and remember for cleanup.
    """
    # 1) Prefer an existing materialized path (no copy)
    p = store.get_materialized_path(run_path, seed_hash)
    if p and p.exists() and Hash().hash_file(p) == seed_hash:
        return p, False

    # 2) Try recorded source_path (the original file), if any
    p2 = store.get_recorded_source_path(run_path, seed_hash)
    if p2 and p2.exists() and Hash().hash_file(p2) == seed_hash:
        return p2, False

    # 3) No local copy → replay the seed into a temporary file (EPHEMERAL)
    prod = store.producer_step_for_data(run_path, seed_hash)
    if not prod:
        raise SystemExit(
            f"restore: seed {seed_hash} missing and no producer step recorded"
        )

    tmpdir = Path(tempfile.mkdtemp(prefix="blase-seed-"))
    tmpfile = tmpdir / f"{seed_hash}.csv"

    # Execute only the seed’s plan, writing to tmpfile; prevent materialization records.
    from .replay import _exec_plan_for_step

    _exec_plan_for_step(
        run_path,
        prod,
        to_path=str(tmpfile),
        backend_override=None,
        seen_steps=seen_steps,
        created_paths=None,
        ephemeral_only=True,
    )

    if not tmpfile.exists() or Hash().hash_file(tmpfile) != seed_hash:
        raise SystemExit("restore: ephemeral seed replay produced wrong bytes")

    if created_paths is not None:
        created_paths.append(tmpfile)

    return tmpfile, True


def _pick_viable_seed(run_path: Path, ins: list[dict]) -> Optional[str]:
    """
    From a list of inputs, pick a seed data hash that is either replayable (has a producer)
    or already materialized. Skip dead seeds.
    """
    seeds = [i["data_hash"] for i in ins if i.get("role") == "seed"]
    if not seeds:
        return None
    for dh in seeds:
        if store.producer_step_for_data(run_path, dh):
            return dh
    for dh in seeds:
        try:
            kind = store.get_data_kind(run_path, dh) or "csv"
            materialize.ensure_local(
                run_path, dh, kind=kind, policy="reuse", to_dir=None
            )
            return dh
        except Exception:
            pass
    return None

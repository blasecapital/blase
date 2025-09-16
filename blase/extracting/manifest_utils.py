import sqlite3
from collections import Counter
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Optional
import json


def build_manifest_descriptor(
    manifest: List[Dict[str, Any]],
    directory: str,
    pattern: str,
    recursive: bool,
    seed: Optional[int],
    root_hash: str,
    hash_mode: str = "content",
) -> Dict[str, Any]:
    """
    Package an image manifest + scan parameters into a compact, serializable descriptor.

    This descriptor is what you persist in CAS (nodes.db `data.metadata_json`)
    for `kind='image.manifest:v1'`. It **does not** include pixel data and avoids
    storing the full per-item metadata to keep the CAS entry small. Downstream
    tools (restore/inspect/CLI) can join back to per-item info via dataset tables.

    Parameters
    ----------
    manifest : list of dict
        Records produced by the header scan (and optionally augmented with hashes),
        each containing fields like:
          - "rel_path", "abs_path", "bytes", "width", "height", "mode",
            "channels", "is_corrupt", "hash_path_mtime", (optional) "hash_content"
        Only a subset is summarized here; the full list stays in memory / dataset tables.
    directory : str
        Root directory scanned. Stored verbatim (absolute/expanded recommended upstream).
    pattern : str
        Glob pattern used (e.g., "**/*.jpg").
    recursive : bool
        Whether the scan recursed into subdirectories.
    seed : int or None
        Shuffle seed used before batching (if any). Logged for reproducibility.
    root_hash : str
        Strong dataset identity (Merkle-like hash over `(rel_path, hash)` per item).
    hash_mode : {"content", "path_mtime_size"}, default "content"
        Which per-item identity the root hash was based on.

    Returns
    -------
    descriptor : dict
        A compact, versioned manifest descriptor with:
          - "type": "image.manifest"
          - "version": "1"
          - "scan": {directory, pattern, recursive, seed, hash_mode}
          - "identity": {root_hash, count, corrupt_count}
          - "stats": {bytes_{min,median,max}, width_{min,median,max}, height_{min,median,max},
                      mode_counts}
          - "samples": {"files_head": [rel_paths...], "files_tail": [rel_paths...], "limit": int}
            (small peek to aid debugging without embedding the entire file list)

    Notes
    -----
    - We intentionally **avoid** embedding the full file list to keep the descriptor small.
      Use `datasets` / `dataset_members` for full membership with order.
    - All numbers are derived from header-only fields; they’re cheap to compute.
    - `mode_counts` reflects Pillow modes seen in headers (e.g., RGB/L/RGBA/CMYK).
    - `corrupt_count` counts items flagged unreadable by header parsing; such items
      should still have byte-level content hashes computed elsewhere if `hash_mode="content"`.

    Examples
    --------
    >>> desc = build_manifest_descriptor(items, "data/imgs", "**/*.jpg", True, 42, root_hash, "content")
    >>> desc["identity"]["count"], desc["identity"]["root_hash"][:8]
    (12873, '9c41f2d0')
    """
    # Pull simple sequences for stats (ignore missing/None)
    widths = [int(r["width"]) for r in manifest if (r.get("width") is not None)]
    heights = [int(r["height"]) for r in manifest if (r.get("height") is not None)]
    sizes = [int(r["bytes"]) for r in manifest if (r.get("bytes") is not None)]
    modes = [str(r["mode"]) for r in manifest if r.get("mode")]

    def _mmmed(xs: List[int]) -> Dict[str, Optional[int]]:
        if not xs:
            return {"min": None, "median": None, "max": None}
        xs_sorted = sorted(xs)
        return {
            "min": xs_sorted[0],
            "median": int(median(xs_sorted)),
            "max": xs_sorted[-1],
        }

    mode_counts = dict(Counter(modes))
    corrupt_count = sum(1 for r in manifest if r.get("is_corrupt"))

    # Small, bounded peek at file list for debugging (not full membership)
    # Keep this tiny to avoid ballooning the CAS entry.
    SAMPLE_LIMIT = 20  # total (split half head / half tail)
    rels = [str(r.get("rel_path", "")) for r in manifest]
    head_count = min(SAMPLE_LIMIT // 2, len(rels))
    tail_count = min(SAMPLE_LIMIT - head_count, max(0, len(rels) - head_count))
    files_head = rels[:head_count]
    files_tail = rels[-tail_count:] if tail_count > 0 else []

    descriptor: Dict[str, Any] = {
        "type": "image.manifest",
        "version": "1",
        "scan": {
            "directory": directory,
            "pattern": pattern,
            "recursive": bool(recursive),
            "seed": int(seed) if isinstance(seed, int) else None,
            "hash_mode": (hash_mode or "content"),
        },
        "identity": {
            "root_hash": root_hash,
            "count": len(manifest),
            "corrupt_count": corrupt_count,
        },
        "stats": {
            "bytes": _mmmed(sizes),
            "width": _mmmed(widths),
            "height": _mmmed(heights),
            "mode_counts": mode_counts,
        },
        "samples": {
            "files_head": files_head,
            "files_tail": files_tail,
            "limit": SAMPLE_LIMIT,
        },
    }
    return descriptor


def ensure_dataset_for_manifest(
    manifest_hash: str,
    manifest: List[Dict[str, Any]],
    tracker,
    cache_member_fields: bool,
    root_hash: str,
) -> str:
    """
    Ensure a datasets row + ordered membership exist for this image manifest.

    This function:
      1) creates/updates a `datasets` row representing the manifest,
      2) registers per-image header metadata as CAS `data` rows
         (kind='image.file.meta', version='1'),
      3) inserts ordered `dataset_members` rows pointing to those per-image `data_hash`es,
      4) returns a stable `dataset_id` derived from the manifest identity.

    Parameters
    ----------
    manifest_hash : str
        CAS id of the manifest descriptor (from `build_manifest_descriptor`).
    manifest : list of dict
        Records from `scan_manifest_headers(...)`, optionally augmented with
        `hash_content`. Must include:
          - "rel_path", "abs_path", "bytes", "width", "height",
            "mode", "channels", "is_corrupt", "hash_path_mtime"
        `hash_content` (if present) will be included in per-image meta.
    tracker : object
        The active step object (e.g., `stream.step`) with:
          - `register_data(kind, version, path_or_bytes, metadata) -> data_hash`
    cache_member_fields : bool
        If True, copy `is_corrupt`, `width`, `height` into `dataset_members`
        for fast CLI health checks (avoids JSON extraction at query time).
    root_hash : str
        Strong dataset identity (Merkle-like root over per-item hashes).

    Returns
    -------
    dataset_id : str
        Stable identifier for this manifest dataset. Derived from `root_hash`:
        e.g., "image.manifest:9c41f2d0..." (prefix + full hash to avoid collisions).

    Notes
    -----
    - Idempotent: re-running will upsert the `datasets` row and de-dup `data`
      via CAS keys (since `register_data` is content-addressed). Membership rows
      are protected by UNIQUE(dataset_id, data_hash); we use INSERT OR IGNORE.
    - Does not store pixel data; per-image `data.metadata_json` contains only
      header/identity fields.
    - If your schema includes a `datasets.root_hash` column, we set it via SQL;
      if not, the root hash is still recorded inside `datasets.metadata_json`.

    """
    if not root_hash:
        raise ValueError("root_hash must be a non-empty string")
    if not manifest_hash:
        raise ValueError("manifest_hash must be a non-empty string")

    # We cannot rely on tracker.conn; use the DB path instead.
    db_path = getattr(tracker, "db", None)
    if not db_path:
        raise RuntimeError("step object must expose a DB path at `tracker.db`")

    dataset_id = f"image.manifest:{root_hash}"
    created_at = datetime.now(timezone.utc).isoformat()

    # Build small metadata payload for the datasets row
    ds_meta = {
        "root_hash": root_hash,
        "manifest_hash": manifest_hash,
        "counts": {"items": len(manifest)},
    }
    ds_meta_json = json.dumps(ds_meta, ensure_ascii=False)

    def table_cols(conn: sqlite3.Connection, table: str) -> set:
        """Probe table columns so we can work with legacy or revised schemas"""
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cur.fetchall()}

    # 1) Upsert datasets row
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        ds_cols = table_cols(conn, "datasets")

        if {"dataset_id", "name", "metadata_json"} <= ds_cols and not (
            {"kind", "created_at", "manifest_hash", "root_hash"} <= ds_cols
        ):
            # Legacy schema: (dataset_id, name, metadata_json)
            conn.execute(
                """
                INSERT INTO datasets (dataset_id, name, metadata_json)
                VALUES (?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                  name = excluded.name,
                  metadata_json = excluded.metadata_json
                """,
                (dataset_id, dataset_id, ds_meta_json),
            )
        else:
            # Revised schema (any superset). Build dynamic INSERT with available columns.
            # Preferred columns:
            desired = [
                ("dataset_id", dataset_id),
                ("name", dataset_id),
                ("kind", "image.manifest:v1"),
                ("created_at", created_at),
                ("manifest_hash", manifest_hash),
                ("root_hash", root_hash),
                ("metadata_json", ds_meta_json),
            ]
            cols = [c for (c, _) in desired if c in ds_cols]
            vals = [v for (c, v) in desired if c in ds_cols]
            placeholders = ",".join(["?"] * len(cols))
            collist = ",".join(cols)
            # Build an UPSERT that updates the same cols on conflict
            setlist = ",".join([f"{c}=excluded.{c}" for c in cols if c != "dataset_id"])
            conn.execute(
                f"""
                INSERT INTO datasets ({collist}) VALUES ({placeholders})
                ON CONFLICT(dataset_id) DO UPDATE SET {setlist}
                """,
                vals,
            )

    # 2) Register per-image meta and 3) insert dataset_members
    dm_cols = table_cols(conn, "dataset_members")
    # Determine which member columns we can fill
    has_role = "role" in dm_cols
    has_ic = "is_corrupt" in dm_cols
    has_w = "width" in dm_cols
    has_h = "height" in dm_cols

    member_rows = []
    for ordinal, rec in enumerate(manifest, 1):
        item_meta = {
            "rel_path": rec.get("rel_path"),
            "abs_path": rec.get("abs_path"),
            "bytes": rec.get("bytes"),
            "mtime": rec.get("mtime"),
            "width": rec.get("width"),
            "height": rec.get("height"),
            "mode": rec.get("mode"),
            "channels": rec.get("channels"),
            "is_corrupt": bool(rec.get("is_corrupt")),
            "hash_path_mtime": rec.get("hash_path_mtime"),
        }
        if rec.get("hash_content"):
            item_meta["hash_content"] = rec["hash_content"]

        # Store descriptor bytes in CAS; also keep metadata in nodes.data for SQL
        payload = json.dumps(item_meta, ensure_ascii=False).encode("utf-8")
        item_hash = tracker.register_data(
            kind="image.file.meta",
            version="1",
            path_or_bytes=payload,
            metadata=item_meta,
        )

        row = {
            "dataset_id": dataset_id,
            "data_hash": item_hash,
            "ordinal": int(ordinal),
        }
        if has_role:
            row["role"] = "item"
        if has_ic:
            row["is_corrupt"] = 1 if rec.get("is_corrupt") else 0
        if has_w:
            row["width"] = int(rec.get("width") or 0)
        if has_h:
            row["height"] = int(rec.get("height") or 0)

        member_rows.append(row)

        # Dynamic INSERT for dataset_members
        with sqlite3.connect(db_path) as conn:
            if member_rows:
                cols = list(member_rows[0].keys())
                placeholders = ",".join(["?"] * len(cols))
                collist = ",".join(cols)
                values = [tuple(r[c] for c in cols) for r in member_rows]
                # Use INSERT OR IGNORE to keep it idempotent (PRIMARY KEY/UNIQUE constraints)
                conn.executemany(
                    f"INSERT OR IGNORE INTO dataset_members ({collist}) VALUES ({placeholders})",
                    values,
                )

    return dataset_id

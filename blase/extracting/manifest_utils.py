import sqlite3
from collections import Counter
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json


def _mmmed(xs: List[int]) -> Dict[str, Optional[int]]:
    if not xs:
        return {"min": None, "median": None, "max": None}
    s = sorted(xs)
    return {"min": s[0], "median": int(median(s)), "max": s[-1]}


def _sample_list(items: List[str], limit: int = 20) -> Dict[str, Any]:
    head_n = min(limit // 2, len(items))
    tail_n = min(limit - head_n, max(0, len(items) - head_n))
    return {
        "files_head": items[:head_n],
        "files_tail": items[-tail_n:] if tail_n > 0 else [],
        "limit": limit,
    }


def build_image_manifest_descriptor(
    manifest: List[Dict[str, Any]],
    *,
    directory: Optional[str],
    pattern: Optional[str],
    recursive: Optional[bool],
    seed: Optional[int],
    root_hash: str,
    hash_mode: str = "content",
) -> Dict[str, Any]:
    widths = [int(r["width"]) for r in manifest if r.get("width") is not None]
    heights = [int(r["height"]) for r in manifest if r.get("height") is not None]
    sizes = [int(r["bytes"]) for r in manifest if r.get("bytes") is not None]
    modes = [str(r["mode"]) for r in manifest if r.get("mode")]
    mode_counts = dict(Counter(modes))
    corrupt_count = sum(1 for r in manifest if r.get("is_corrupt"))

    rels = [str(r.get("rel_path", "")) for r in manifest]

    return {
        "type": "image.manifest",
        "version": "1",
        "scan": {
            "directory": str(Path(directory).expanduser().resolve())
            if directory
            else None,
            "pattern": pattern or "",
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
        "samples": _sample_list(rels, 20),
    }


def build_table_manifest_descriptor(
    manifest: Dict[str, Any],
    *,
    deterministic: Optional[bool],
    root_hash: str,
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = list(manifest.get("entries") or [])
    sources = sorted(str(Path(s).resolve()) for s in (manifest.get("sources") or []))

    rows_per_file = [int(e.get("rows") or 0) for e in entries]
    rgs_per_file = [int(e.get("row_groups") or 0) for e in entries]
    sizes_bytes = [int(e.get("size")) for e in entries if e.get("size") is not None]
    file_paths = [str(e.get("path", "")) for e in entries]

    return {
        "type": "table.manifest",
        "version": "1",
        "scan": {
            "kind": manifest.get("kind"),
            "sources": sources,
            "deterministic": bool(deterministic) if deterministic is not None else None,
            "projected_columns": list(manifest.get("projected_columns") or []),
            "filters_present": bool(manifest.get("filters_present")),
            "filters_repr": manifest.get("filters_repr"),
        },
        "identity": {
            "root_hash": root_hash,
            "files": len(entries),
            "rows": int(manifest.get("rows") or 0),
            "row_groups": int(manifest.get("row_groups") or 0),
            "schema_fp": manifest.get("schema_fp"),
        },
        "schema": {
            "columns": list(manifest.get("columns") or []),
        },
        "stats": {
            "rows_per_file": _mmmed(rows_per_file),
            "row_groups_per_file": _mmmed(rgs_per_file),
            "size_bytes": _mmmed(sizes_bytes),
        },
        "samples": _sample_list(file_paths, 20),
    }


def build_manifest_descriptor(
    manifest: Union[List[Dict[str, Any]], Dict[str, Any]],
    *,
    directory: Optional[str] = None,
    pattern: Optional[str] = None,
    recursive: Optional[bool] = None,
    seed: Optional[int] = None,
    hash_mode: str = "content",
    root_hash: str,
    deterministic: Optional[bool] = None,
) -> Dict[str, Any]:
    if isinstance(manifest, dict):
        return build_table_manifest_descriptor(
            manifest, deterministic=deterministic, root_hash=root_hash
        )
    return build_image_manifest_descriptor(
        manifest,
        directory=directory,
        pattern=pattern,
        recursive=recursive,
        seed=seed,
        root_hash=root_hash,
        hash_mode=hash_mode,
    )


# ---------- shared DB helpers ----------
def _table_cols(conn: sqlite3.Connection, table: str) -> set:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _ensure_datasets_row(
    db_path: str,
    *,
    dataset_id: str,
    kind_tag: str,  # e.g., "image.manifest:v1" or "table.manifest:v1"
    manifest_hash: str,
    root_hash: str,
    counts: Dict[str, int],  # {"items": N} and optionally {"rows": M, "row_groups": K}
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    ds_meta = {
        "root_hash": root_hash,
        "manifest_hash": manifest_hash,
        "counts": counts,
    }
    ds_meta_json = json.dumps(ds_meta, ensure_ascii=False)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        ds_cols = _table_cols(conn, "datasets")

        # Legacy schema: (dataset_id, name, metadata_json)
        if {"dataset_id", "name", "metadata_json"} <= ds_cols and not (
            {"kind", "created_at", "manifest_hash", "root_hash"} <= ds_cols
        ):
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
            return

        # Revised or superset schema
        desired = [
            ("dataset_id", dataset_id),
            ("name", dataset_id),
            ("kind", kind_tag),
            ("created_at", created_at),
            ("manifest_hash", manifest_hash),
            ("root_hash", root_hash),
            ("metadata_json", ds_meta_json),
        ]
        cols = [c for (c, _) in desired if c in ds_cols]
        vals = [v for (c, v) in desired if c in ds_cols]
        placeholders = ",".join(["?"] * len(cols))
        collist = ",".join(cols)
        setlist = ",".join([f"{c}=excluded.{c}" for c in cols if c != "dataset_id"])

        conn.execute(
            f"""
            INSERT INTO datasets ({collist}) VALUES ({placeholders})
            ON CONFLICT(dataset_id) DO UPDATE SET {setlist}
            """,
            vals,
        )


def _insert_dataset_members(
    db_path: str,
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ",".join(["?"] * len(cols))
    collist = ",".join(cols)
    values = [tuple(r[c] for c in cols) for r in rows]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            f"INSERT OR IGNORE INTO dataset_members ({collist}) VALUES ({placeholders})",
            values,
        )


# ---------- CAS registration helpers ----------
def _register_image_member(tracker, rec: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    meta = {
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
        meta["hash_content"] = rec["hash_content"]

    payload = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    data_hash = tracker.register_data(
        kind="image.file.meta",
        version="1",
        path_or_bytes=payload,
        metadata=meta,
    )
    return data_hash, meta


def _register_table_member(tracker, ent: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    meta = {
        "path": ent.get("path"),
        "rows": int(ent.get("rows") or 0),
        "row_groups": int(ent.get("row_groups") or 0),
        "size": ent.get("size"),
        "mtime": ent.get("mtime"),
    }
    payload = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    data_hash = tracker.register_data(
        kind="table.file.meta",
        version="1",
        path_or_bytes=payload,
        metadata=meta,
    )
    return data_hash, meta


# ---------- main dispatcher ----------
def ensure_dataset_for_manifest(
    manifest_hash: str,
    manifest: Union[List[Dict[str, Any]], Dict[str, Any]],
    tracker,
    cache_member_fields: bool,
    root_hash: str,
) -> str:
    """
    Generalized:
      - If manifest is list[dict] → image dataset.
      - If manifest is dict      → table/parquet dataset.

    Returns dataset_id.
    """
    if not root_hash:
        raise ValueError("root_hash must be a non-empty string")
    if not manifest_hash:
        raise ValueError("manifest_hash must be a non-empty string")

    db_path = getattr(tracker, "db", None)
    if not db_path:
        raise RuntimeError("tracker must expose a DB path at `tracker.db`")

    # Image branch
    if isinstance(manifest, list):
        dataset_id = f"image.manifest:{root_hash}"

        # upsert datasets row
        counts = {"items": len(manifest)}
        _ensure_datasets_row(
            db_path,
            dataset_id=dataset_id,
            kind_tag="image.manifest:v1",
            manifest_hash=manifest_hash,
            root_hash=root_hash,
            counts=counts,
        )

        # dataset_members shape discovery
        with sqlite3.connect(db_path) as conn:
            dm_cols = _table_cols(conn, "dataset_members")

        has_role = "role" in dm_cols
        has_ic = "is_corrupt" in dm_cols
        has_w = "width" in dm_cols
        has_h = "height" in dm_cols

        rows: List[Dict[str, Any]] = []
        for ordinal, rec in enumerate(manifest, 1):
            data_hash, meta = _register_image_member(tracker, rec)
            row = {
                "dataset_id": dataset_id,
                "data_hash": data_hash,
                "ordinal": int(ordinal),
            }
            if has_role:
                row["role"] = "item"
            if cache_member_fields:
                if has_ic:
                    row["is_corrupt"] = 1 if rec.get("is_corrupt") else 0
                if has_w:
                    row["width"] = int(rec.get("width") or 0)
                if has_h:
                    row["height"] = int(rec.get("height") or 0)
            rows.append(row)

        _insert_dataset_members(db_path, rows)
        return dataset_id

    # Table/Parquet branch
    if isinstance(manifest, dict):
        entries: List[Dict[str, Any]] = list(manifest.get("entries") or [])
        dataset_id = f"table.manifest:{root_hash}"

        counts = {
            "items": len(entries),
            "rows": int(manifest.get("rows") or 0),
            "row_groups": int(manifest.get("row_groups") or 0),
        }
        _ensure_datasets_row(
            db_path,
            dataset_id=dataset_id,
            kind_tag="table.manifest:v1",
            manifest_hash=manifest_hash,
            root_hash=root_hash,
            counts=counts,
        )

        # dataset_members shape discovery
        with sqlite3.connect(db_path) as conn:
            dm_cols = _table_cols(conn, "dataset_members")

        has_role = "role" in dm_cols
        has_rows = "rows" in dm_cols
        has_rgs = "row_groups" in dm_cols
        has_size = "size" in dm_cols

        rows_out: List[Dict[str, Any]] = []
        for ordinal, ent in enumerate(entries, 1):
            data_hash, meta = _register_table_member(tracker, ent)
            row = {
                "dataset_id": dataset_id,
                "data_hash": data_hash,
                "ordinal": int(ordinal),
            }
            if has_role:
                row["role"] = "file"
            if cache_member_fields:
                if has_rows:
                    row["rows"] = int(ent.get("rows") or 0)
                if has_rgs:
                    row["row_groups"] = int(ent.get("row_groups") or 0)
                if has_size and ent.get("size") is not None:
                    row["size"] = int(ent.get("size"))
            rows_out.append(row)

        _insert_dataset_members(db_path, rows_out)
        return dataset_id

    raise ValueError("Unsupported manifest type; expected list[dict] or dict.")

from typing import Union, Sequence, Optional, List, Any, Dict, Tuple, Literal, Generator
from pathlib import Path

from blase.utils.hashing import Hash


def _normalize_source_list(
    source: Union[str, Path, Sequence[Union[str, Path]]],
    pattern: str = "**/*.parquet",
    recursive: bool = True,
    deterministic: bool = True,
    limit_files: Optional[int] = None,
) -> List[Path]:
    """
    Normalize source(s) to a list of Parquet file paths.

    - source: file path, directory path, or list of these
    - pattern: glob pattern (default '**/*.parquet')
    - recursive: if False, run Path.glob(pattern) instead of rglob
    - deterministic: sort the final list for stable ordering
    - limit_files: optional maximum number of files to return
    """
    paths: List[Path] = []

    def _expand_dir(p: Path) -> List[Path]:
        if recursive:
            return list(p.rglob(pattern))
        else:
            return list(p.glob(pattern))

    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            paths = _expand_dir(p)
        else:
            paths = [p]
    else:
        for s in source:
            p = Path(s)
            if p.is_dir():
                paths.extend(_expand_dir(p))
            else:
                paths.append(p)

    if deterministic:
        paths.sort(key=lambda q: str(q))

    if isinstance(limit_files, int):
        paths = paths[: max(0, limit_files)]

    return paths


def _scan_parquet_manifest_stub(
    sources: List[Path],
    fmt: str,
    columns: Optional[Sequence[str]],
    filters: Any,
    schema: Any,
) -> Dict[str, Any]:
    """
    Header-only scan for Parquet/Arrow files.
    - No data pages read.
    - Returns per-file entries, totals, and a schema fingerprint.
    """
    kind: str
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        import pyarrow.dataset as ds  # type: ignore
    except ImportError:
        print("Error: pyarrow modules could not be imported.")

    def _schema_fp_arrow(s: "pa.Schema") -> str:
        # stable-ish fingerprint from names+types+metadata keys
        hasher = Hash()
        parts = []
        for f in s:
            parts.append(f"{f.name}:{str(f.type)}")
        md = s.metadata or {}
        mkeys = sorted(k.decode("utf8") for k in md.keys())
        parts.append("md:" + "|".join(mkeys))
        return "schema:" + hasher.hash_bytes("|".join(parts).encode("utf8"))

    def _file_stats(p: Path) -> Dict[str, Optional[int]]:
        try:
            st = p.stat()
            return {"size": int(st.st_size), "mtime": int(st.st_mtime)}
        except OSError:
            return {"size": None, "mtime": None}

    entries: List[Dict[str, Any]] = []
    total_rows = 0
    total_rgs = 0
    schema_fp: Optional[str] = None
    all_columns: Optional[List[str]] = None

    # Decide path: single-file vs dataset
    files = [p for p in sources if p.is_file()]
    dirs = [p for p in sources if p.is_dir()]
    if dirs:
        kind = "dataset"
        # Dataset scan: schema + per-file metadata via fragments
        dataset = ds.dataset([str(p) for p in (dirs + files)], format="parquet")
        files = sorted(dataset.files)
        sch = dataset.schema
        schema_fp = _schema_fp_arrow(sch)
        all_columns = list(sch.names)

        # enumerate fragments without materializing data
        for f in files:
            # file-level info
            p = Path(f)
            # row-group metadata
            try:
                md = (
                    f.inspect()
                )  # FragmentScanOptions -> FileFragment.inspect() → FileInfo
            except Exception:
                md = None
            # use ParquetFile when possible to get row groups and rows
            rg_count = 0
            row_count = 0
            if f.exists():
                try:
                    pf = pq.ParquetFile(f)
                    rg_count = pf.metadata.num_row_groups
                    row_count = pf.metadata.num_rows
                except Exception:
                    pass
            stats = _file_stats(p)
            entries.append(
                {
                    "path": f,
                    "rows": int(row_count),
                    "row_groups": int(rg_count),
                    "size": stats["size"],
                    "mtime": stats["mtime"],
                }
            )
            total_rows += row_count
            total_rgs += rg_count

    else:
        # Single or many explicit files → treat as 'parquet' unless user forces 'feather'
        kind = (
            "parquet"
            if fmt in ("auto", "parquet")
            else ("feather" if fmt == "feather" else "parquet")
        )
        # Collect schema from first readable file
        first_schema = None
        for p in files:
            try:
                if kind == "feather" or p.suffix.lower() in (".feather", ".ft"):
                    ipc = pa.ipc.open_file(str(p))  # header only
                    first_schema = ipc.schema
                else:
                    pf = pq.ParquetFile(p)
                    first_schema = pf.schema_arrow
                break
            except Exception:
                continue
        if first_schema is None:
            raise RuntimeError("first_schema is None")

        schema_fp = _schema_fp_arrow(first_schema)
        all_columns = list(first_schema.names)

        # Per-file fast metadata
        for p in files:
            rg_count = 0
            row_count = 0
            try:
                if kind == "feather" or p.suffix.lower() in (".feather", ".ft"):
                    ipc = pa.ipc.open_file(str(p))
                    row_count = int(ipc.num_rows) if hasattr(ipc, "num_rows") else 0
                    rg_count = 1
                else:
                    pf = pq.ParquetFile(p)
                    md = pf.metadata  # no data read
                    rg_count = md.num_row_groups
                    row_count = md.num_rows
            except Exception:
                pass
            stats = _file_stats(p)
            entries.append(
                {
                    "path": str(p),
                    "rows": int(row_count),
                    "row_groups": int(rg_count),
                    "size": stats["size"],
                    "mtime": stats["mtime"],
                }
            )
            total_rows += int(row_count)
            total_rgs += int(rg_count)

    return {
        "kind": kind,
        "sources": [str(p) for p in sources],
        "rows": int(total_rows),
        "row_groups": int(total_rgs),
        "schema_fp": schema_fp,
        "columns": all_columns,
        "projected_columns": list(columns) if columns else None,
        "filters_present": bool(filters),
        "entries": entries,
    }


def _ensure_parquet_content_hashes(
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = list(manifest.get("entries") or [])

    for ent in entries:
        if ent.get("hash_content"):
            continue
        p = ent.get("path")
        if not p:
            ent["hash_content"] = None
            continue
        try:
            h = Hash().hash_file(Path(p))
        except Exception:
            h = None
        ent["hash_content"] = h
    manifest["entries"] = entries
    return manifest


def _compute_manifest_root_hash_stub(manifest: Dict[str, Any]) -> str:
    """
    Deterministic identity for a Parquet/Feather manifest without reading data pages.
    Uses schema fp, normalized sources, projected columns, filters, and per-file facts.
    """
    # Normalize sources and entries deterministically
    srcs = [str(Path(s).resolve()) for s in (manifest.get("sources") or [])]
    srcs.sort()

    entries = manifest.get("entries") or []
    entries_sorted = sorted(entries, key=lambda e: str(e.get("path", "")))
    entry_tuples: List[Any] = []
    for e in entries_sorted:
        entry_tuples.append(
            (
                str(Path(str(e.get("path", ""))).resolve()),
                int(e.get("rows") or 0),
                int(e.get("row_groups") or 0),
                int(e.get("size") or -1),
                int(e.get("mtime") or -1),
            )
        )

    identity = {
        "type": "table.manifest:v1",
        "kind": manifest.get("kind"),
        "schema_fp": manifest.get("schema_fp"),
        "sources": srcs,
        "columns": list(manifest.get("columns") or []),  # discovered columns
        "projected_columns": list(manifest.get("projected_columns") or []),
        # prefer a canonical string; else boolean presence
        "filters": manifest.get("filters_repr")
        or (True if manifest.get("filters_present") else None),
        "entries": entry_tuples,  # per-file facts
    }

    h = Hash()
    return h.hash_object(identity)


def _plan_parquet_batches_stub(
    manifest: Dict[str, Any],
    *,
    mode: Literal["table", "images"] = "table",
    batch_size: Optional[int] = None,
    max_rows: Optional[int] = None,
    target_batch_bytes: Optional[int] = None,
    safety_margin: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Plan Parquet reads as row-group aligned batches by default, with a bytes-aware
    greedy packer when batch_size is not provided. For images mode, ensures integer
    row counts. Supports dataset or single-file manifests.

    Manifest expectations (best effort):
      manifest = {
        "kind": "dataset" | "parquet" | "feather",
        "rows": int,
        "row_groups": int,
        "entries": [
           {
             "path": str,
             "rows": int,
             "row_groups": int,
             # Optional detail for better estimates:
             # "rg_rows": List[int],
             # "rg_uncompressed_bytes": List[int],
             # "size": int (file size in bytes),
           }, ...
        ],
        "columns": Optional[List[str]],
      }
    """
    # ---- helpers -------------------------------------------------------------

    def _default_target_bytes() -> int:
        # 70% of available RAM, else 256 MiB
        try:
            import psutil  # type: ignore

            avail = int(getattr(psutil.virtual_memory(), "available", 0))
            if avail > 0:
                return max(int(avail * 0.70), 64 * 1024 * 1024)
        except Exception:
            pass
        return 256 * 1024 * 1024

    def _inflate_ratio() -> float:
        # Conservative inflate from file-size to in-memory. Higher for images.
        return 1.8 if mode == "images" else 1.2

    def _rg_rows_list(entry: Dict[str, Any]) -> List[int]:
        rg = int(entry.get("row_groups") or 0)
        rows = int(entry.get("rows") or 0)
        if rg <= 0:
            return []
        if (
            "rg_rows" in entry
            and isinstance(entry["rg_rows"], list)
            and entry["rg_rows"]
        ):
            return [int(x or 0) for x in entry["rg_rows"]]
        # even split, last takes remainder
        base = rows // rg
        rem = rows - base * rg
        out = [base] * rg
        for i in range(rem):
            out[i] += 1
        return out

    def _rg_est_bytes_list(entry: Dict[str, Any], rg_rows: List[int]) -> List[int]:
        if "rg_uncompressed_bytes" in entry and isinstance(
            entry["rg_uncompressed_bytes"], list
        ):
            vals = [int(x or 0) for x in entry["rg_uncompressed_bytes"]]
            if len(vals) == len(rg_rows):
                return vals
        # fallback: file size per row * rg_rows * inflate
        size = int(entry.get("size") or 0)
        total_rows = max(int(entry.get("rows") or 0), 1)
        per_row = (
            (size / total_rows) if size > 0 else (64000 if mode == "images" else 8192)
        )
        inf = _inflate_ratio()
        return [int(max(1, per_row) * r * inf) for r in rg_rows]

    def _cap_bytes(user_cap: Optional[int]) -> int:
        cap = (
            int(user_cap)
            if (isinstance(user_cap, int) and user_cap > 0)
            else _default_target_bytes()
        )
        cap = int(cap * (1.0 - float(safety_margin)))
        return max(cap, 1)

    # ---- build row-group stream ---------------------------------------------

    entries = manifest.get("entries") or []
    rg_stream: List[Dict[str, Any]] = []  # each: {path, rg_index, rows, est_bytes}

    for ent in entries:
        path = ent.get("path")
        rg_rows = _rg_rows_list(ent)
        ests = _rg_est_bytes_list(ent, rg_rows) if rg_rows else []
        for i, rows in enumerate(rg_rows):
            rg_stream.append(
                {
                    "path": path,
                    "rg_index": i,
                    "rows": int(rows),
                    "est_bytes": int(ests[i] if i < len(ests) else 0),
                }
            )

    # If no RG info, synthesize a single chunk per file.
    if not rg_stream and entries:
        for ent in entries:
            rows = int(ent.get("rows") or 0)
            rg_stream.append(
                {
                    "path": ent.get("path"),
                    "rg_index": 0,
                    "rows": rows,
                    "est_bytes": int(
                        (
                            int(ent.get("size") or 0)
                            or rows * (64000 if mode == "images" else 8192)
                        )
                        * _inflate_ratio()
                    ),
                }
            )

    total_rows = int(manifest.get("rows") or sum(x["rows"] for x in rg_stream))
    rows_budget = (
        int(max_rows) if isinstance(max_rows, int) and max_rows > 0 else total_rows
    )
    rows_remaining = max(rows_budget, 0)

    plans: List[Dict[str, Any]] = []
    if rows_remaining <= 0:
        return [
            {
                "fragment": None,
                "row_group": None,
                "offset": 0,
                "count": 0,
                "is_last": True,
            }
        ]

    # ---- manual by count -----------------------------------------------------

    if isinstance(batch_size, int) and batch_size > 0:
        # Walk RGs, slicing the last RG in a batch if needed.
        cur_count = 0
        cur_batch: List[
            Tuple[str, int, int, int]
        ] = []  # (path, rg_index, offset, count)

        def _flush(is_last: bool = False):
            nonlocal cur_batch, cur_count
            if not cur_batch:
                return
            for path, rg_idx, offset, count in cur_batch:
                plans.append(
                    {
                        "fragment": path,
                        "row_group": rg_idx,
                        "offset": offset,
                        "count": count,
                        "is_last": False,  # set final True after planning
                    }
                )
            cur_batch, cur_count = [], 0

        for rg in rg_stream:
            if rows_remaining <= 0:
                break
            take_from_rg = min(rg["rows"], rows_remaining)
            pos = 0
            while take_from_rg > 0:
                room = batch_size - cur_count
                if room <= 0:
                    _flush(is_last=False)
                    room = batch_size
                n = min(room, take_from_rg)
                cur_batch.append((rg["path"], rg["rg_index"], pos, n))
                cur_count += n
                pos += n
                take_from_rg -= n
                rows_remaining -= n
                if cur_count >= batch_size:
                    _flush(is_last=False)

        _flush(is_last=True)
        if plans:
            for b in plans[:-1]:
                b["is_last"] = False
            plans[-1]["is_last"] = True
        return plans

    # ---- auto bytes-aware greedy packing ------------------------------------

    cap = _cap_bytes(target_batch_bytes)
    cur_est = 0
    cur_rows = 0
    cur_parts: List[Tuple[str, int, int, int]] = []  # (path, rg_index, offset, count)

    def flush(is_last: bool = False):
        nonlocal cur_est, cur_rows, cur_parts
        if not cur_parts:
            return
        for path, rg_idx, offset, count in cur_parts:
            plans.append(
                {
                    "fragment": path,
                    "row_group": rg_idx,
                    "offset": offset,
                    "count": count,
                    "is_last": False,  # set final True after planning
                }
            )
        cur_est = 0
        cur_rows = 0
        cur_parts = []

    for rg in rg_stream:
        if rows_remaining <= 0:
            break

        # If entire RG would exceed remaining rows, slice it.
        rows_here = min(rg["rows"], rows_remaining)

        # Greedy pack by RG. Do not split by bytes unless batch is empty and RG is huge.
        would_exceed = (cur_est + rg["est_bytes"]) > cap if cur_parts else False

        if cur_parts and would_exceed:
            flush(is_last=False)

        # Ensure progress: if empty and RG est > cap, take RG as its own batch.
        if not cur_parts and rg["est_bytes"] > cap:
            take = rows_here
            cur_parts.append((rg["path"], rg["rg_index"], 0, take))
            cur_est += rg["est_bytes"]
            cur_rows += take
            rows_remaining -= take
            flush(is_last=False)
            continue

        # Add whole RG, possibly with row slicing for max_rows limit.
        take = rows_here
        cur_parts.append((rg["path"], rg["rg_index"], 0, take))
        cur_est += rg["est_bytes"]
        cur_rows += take
        rows_remaining -= take

        # If cap reached, flush.
        if cur_est >= cap:
            flush(is_last=False)

    flush(is_last=True)

    if not plans:
        plans = [
            {
                "fragment": None,
                "row_group": None,
                "offset": 0,
                "count": 0,
                "is_last": True,
            }
        ]
    else:
        for b in plans[:-1]:
            b["is_last"] = False
        plans[-1]["is_last"] = True

    return plans


def _read_parquet_table_stub(
    manifest: Dict[str, Any],
    plan: Dict[str, Any],
    return_type: str,
    columns: Optional[Sequence[str]],
    *,
    mode: Literal["table", "images"] = "table",
    bytes_col: str = "img_bytes",
    dims_cols: Tuple[str, str, str] = ("height", "width", "channels"),
    label_col: Optional[str] = "label",
    path_col: Optional[str] = "path",
    use_threads: bool = True,
    memory_map: bool = True,
    to_pandas_kwargs: Optional[dict] = None,
) -> Any:
    """
    Try to read the exact (fragment,row_group,offset,count) slice with Arrow.
    If Arrow isn't available, return a shaped stub for downstream wiring.
    In images mode, always include the required image columns in the projection.
    """
    # ---- projection ---------------------------------------------------------
    proj = list(columns) if columns else None
    if mode == "images":
        req = [bytes_col, *dims_cols]
        if label_col:
            req.append(label_col)
        if path_col:
            req.append(path_col)
        proj = sorted(set(req) if proj is None else set(proj).union(req))

    # ---- Arrow path ---------------------------------------------------------
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        frag = plan.get("fragment")
        rg = plan.get("row_group")
        off = int(plan.get("offset", 0))
        cnt = int(plan.get("count", 0))

        if not frag or cnt == 0:
            # return empty table/dataframe with projected columns
            cols = proj or (manifest.get("columns") or [])
            if return_type == "pandas":
                import pandas as pd  # type: ignore

                return pd.DataFrame({c: [] for c in cols})
            return pa.table({c: pa.array([], type=pa.null()) for c in cols})

        pf = pq.ParquetFile(frag, memory_map=bool(memory_map))

        if rg is None:
            tbl = pf.read(columns=proj, use_threads=use_threads)
        else:
            tbl = pf.read_row_group(int(rg), columns=proj, use_threads=use_threads)

        if off or cnt:
            tbl = tbl.slice(off, cnt)

        if return_type == "pandas":
            return tbl.to_pandas(**(to_pandas_kwargs or {}))
        return tbl

    except Exception:
        # ---- stub fallback ---------------------------------------------------
        count = int(plan.get("count", 0))
        cols = list(proj or (manifest.get("columns") or []))
        if return_type == "pandas":
            return {"__kind__": "pandas.DataFrame", "rows": count, "columns": cols}
        return {"__kind__": "pyarrow.Table", "rows": count, "columns": cols}


def _compute_batch_hash_stub(root_hash: str, plan: Dict[str, Any]) -> str:
    """
    Stable identity for a single Parquet batch (one file + one row-group slice).

    plan keys expected:
      - fragment: str | None   # file path
      - row_group: int | None  # 0 if feather/single-chunk
      - offset: int            # row offset within the selected row-group
      - count: int             # rows to read
    """
    if not isinstance(root_hash, str) or not root_hash:
        raise ValueError("root_hash must be a non-empty string")

    frag_raw: Optional[str] = plan.get("fragment")
    frag = str(Path(frag_raw).resolve()) if frag_raw else ""
    rg = plan.get("row_group")
    offset = int(plan.get("offset", 0))
    count = int(plan.get("count", 0))

    if count < 0 or offset < 0:
        raise ValueError("offset and count must be non-negative")
    if frag == "" and rg is not None:
        raise ValueError("row_group provided without a fragment path")

    identity = {
        "type": "table.batch:v1",
        "root_hash": root_hash,
        "fragment": frag,
        "row_group": int(rg) if rg is not None else None,
        "offset": offset,
        "count": count,
    }

    return Hash().hash_object(identity)


def _auto_detect_image_cols(
    table_like: Any,
    bytes_col: Optional[str],
    dims_cols: Tuple[str, str, str],
    label_col: Optional[str],
    path_col: Optional[str],
) -> Tuple[str, Tuple[str, str, str], Optional[str], Optional[str]]:
    """
    Return effective (bytes_col, (height,width,channels), label_col, path_col).
    Works with pyarrow.Table or pandas.DataFrame. Raises ValueError if required cols missing.
    """
    # --- discover available column names ---
    if hasattr(table_like, "column_names"):  # pyarrow.Table
        available: Sequence[str] = list(table_like.column_names)
    elif hasattr(table_like, "columns"):  # pandas.DataFrame
        available = [str(c) for c in list(table_like.columns)]
    elif isinstance(table_like, dict) and "columns" in table_like:
        available = list(table_like["columns"])
    else:
        available = []

    av_set = set(available)
    av_lower = {c.lower(): c for c in available}

    def _pick(requested: Optional[str], aliases: Sequence[str]) -> Optional[str]:
        # exact first
        if requested and requested in av_set:
            return requested
        # case-insensitive exact
        if requested and requested.lower() in av_lower:
            return av_lower[requested.lower()]
        # alias search
        for a in aliases:
            if a in av_set:
                return a
            if a.lower() in av_lower:
                return av_lower[a.lower()]
        return None

    # alias pools
    bytes_aliases = ["img_bytes", "image", "image_bytes", "bytes", "data", "content"]
    h_aliases = ["height", "h"]
    w_aliases = ["width", "w"]
    c_aliases = ["channels", "ch", "c", "n_channels"]
    label_aliases = ["label", "labels", "y", "target", "class", "category"]
    path_aliases = ["path", "filepath", "file_path", "rel_path", "source_rel_path"]

    # resolve bytes column
    eff_bytes = _pick(bytes_col, bytes_aliases)
    if not eff_bytes:
        raise ValueError(
            f"images mode requires an image-bytes column. Searched: "
            f"{[bytes_col] if bytes_col else [] + bytes_aliases}. Available: {available}"
        )

    # resolve dims
    req_h, req_w, req_c = dims_cols
    eff_h = _pick(req_h, h_aliases)
    eff_w = _pick(req_w, w_aliases)
    eff_c = _pick(req_c, c_aliases)
    missing = [
        name
        for name, eff in zip(["height", "width", "channels"], [eff_h, eff_w, eff_c])
        if eff is None
    ]
    if missing:
        raise ValueError(
            f"missing required image dimension columns: {missing}. "
            f"Searched aliases h={h_aliases}, w={w_aliases}, c={c_aliases}. Available: {available}"
        )

    # optional label/path
    eff_label = (
        _pick(label_col, label_aliases)
        if (
            label_col is None
            or label_col in av_set
            or (label_col and label_col.lower() in av_lower)
        )
        else None
    )
    if eff_label is None and label_col in (None,):
        eff_label = _pick(None, label_aliases)

    eff_path = (
        _pick(path_col, path_aliases)
        if (
            path_col is None
            or path_col in av_set
            or (path_col and path_col.lower() in av_lower)
        )
        else None
    )
    if eff_path is None and path_col in (None,):
        eff_path = _pick(None, path_aliases)

    return eff_bytes, (eff_h, eff_w, eff_c), eff_label, eff_path


def _validate_image_schema(
    table_like: Any,
    bytes_col: str,
    dims_cols: Tuple[str, str, str],
    label_col: Optional[str],
    path_col: Optional[str],
) -> None:
    """
    Validate required columns exist and have compatible types.
    Supports pyarrow.Table, pandas.DataFrame, and a minimal dict stub { 'columns': [...]}.
    Raises ValueError on mismatch.
    """
    try:
        import pyarrow as pa  # type: ignore
    except Exception:
        pa = None  # type: ignore

    # --- discover available columns ---
    if hasattr(table_like, "column_names"):  # pyarrow.Table
        available: Sequence[str] = list(table_like.column_names)

        def get_type(col):
            return table_like.schema.field(col).type

        kind = "arrow"

    elif hasattr(table_like, "columns"):  # pandas.DataFrame
        available = [str(c) for c in list(table_like.columns)]

        def get_type(col):
            return table_like.dtypes[col]  # pandas dtype

        kind = "pandas"

    elif isinstance(table_like, dict) and "columns" in table_like:
        available = list(table_like["columns"])

        def get_type(col):
            return None

        kind = "stub"

    else:
        raise ValueError(
            "Unsupported table_like; expected pyarrow.Table, pandas.DataFrame, or stub dict."
        )

    def _ensure_present(cols: Sequence[Optional[str]], names: Sequence[str]) -> None:
        missing = [n for n, c in zip(names, cols) if not c or c not in available]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. Available: {available}"
            )

    # --- presence checks ---
    h_col, w_col, c_col = dims_cols
    _ensure_present(
        [bytes_col, h_col, w_col, c_col], ["bytes_col", "height", "width", "channels"]
    )

    # --- type helpers ---
    def _is_int_like(t) -> bool:
        if kind == "arrow":
            assert pa is not None
            return pa.types.is_integer(t)
        if kind == "pandas":
            # allow any integer subtype; also accept nullable Int* dtypes
            return str(t).startswith("int") or str(t).startswith("Int")
        return True  # stub: skip strict typing

    def _is_binary_like(t) -> bool:
        if kind == "arrow":
            assert pa is not None
            return pa.types.is_binary(t) or pa.types.is_large_binary(t)
        if kind == "pandas":
            # common storage for bytes is 'object'; cannot strictly enforce
            return str(t) in ("object", "bytes", "|S", "O")
        return True  # stub

    def _is_string_like(t) -> bool:
        if kind == "arrow":
            assert pa is not None
            return pa.types.is_string(t) or pa.types.is_large_string(t)
        if kind == "pandas":
            return str(t).startswith(("string", "object"))
        return True

    # --- type checks (best-effort per backend) ---
    b_t = get_type(bytes_col)
    h_t = get_type(h_col)
    w_t = get_type(w_col)
    c_t = get_type(c_col)

    if not _is_binary_like(b_t):
        raise ValueError(f"{bytes_col} must be binary-like; got {b_t} ({kind}).")
    if not _is_int_like(h_t):
        raise ValueError(f"{h_col} must be integer-like; got {h_t} ({kind}).")
    if not _is_int_like(w_t):
        raise ValueError(f"{w_col} must be integer-like; got {w_t} ({kind}).")
    if not _is_int_like(c_t):
        raise ValueError(f"{c_col} must be integer-like; got {c_t} ({kind}).")

    if label_col and label_col in available:
        # label can be anything; no strict check. If arrow and not null type, allow primitive/list/string
        if kind == "arrow" and pa is not None:
            lt = get_type(label_col)
            ok = (
                pa.types.is_null(lt)
                or pa.types.is_integer(lt)
                or pa.types.is_floating(lt)
                or pa.types.is_string(lt)
                or pa.types.is_large_string(lt)
                or pa.types.is_boolean(lt)
                or pa.types.is_list(lt)
                or pa.types.is_dictionary(lt)
            )
            if not ok:
                raise ValueError(f"{label_col} has unsupported Arrow type {lt}.")

    if path_col and path_col in available:
        pt = get_type(path_col)
        if not _is_string_like(pt):
            raise ValueError(f"{path_col} must be string-like; got {pt} ({kind}).")

    # passed
    return


def _iter_images_from_table(
    table_like: Any,
    return_type: str,
    bytes_col: str,
    dims_cols: Tuple[str, str, str],
    label_col: Optional[str],
    path_col: Optional[str],
    decode: Optional[str],
    images_return: str,
) -> Generator[Tuple[List[Any], Optional[List[Any]], List[Optional[str]]], None, None]:
    """
    Yield one image per row as (images_list, labels_list|None, paths_list).
    Works with pyarrow.Table, pandas.DataFrame, or a minimal stub { 'columns': [...], ... }.
    Decoding path: bytes -> np (optional) -> pil/tensor (optional).
    """

    # -------- helpers --------
    def _to_lists_arrow(tbl, col) -> List[Any]:
        return tbl[col].combine_chunks().to_pylist()

    def _to_lists_pandas(df, col) -> List[Any]:
        return df[col].tolist()

    def _ensure_np_import():
        import importlib

        m = importlib.import_module("numpy")
        return m

    def _decode_to_np(
        img_bytes: bytes, h: Optional[int], w: Optional[int], c: Optional[int]
    ):
        """
        Best-effort bytes->np.uint8[:,:,:].
        Prefer PIL if present; fallback to OpenCV if present; otherwise raw decode error.
        h,w,c are advisory only; not required.
        """
        # Try PIL first
        try:
            from PIL import Image
            import numpy as _np

            im = Image.open(io.BytesIO(img_bytes))  # type: ignore[name-defined]
            im = im.convert("RGB")
            return _np.array(im, dtype=_np.uint8)
        except Exception:
            pass
        # Try OpenCV
        try:
            import numpy as _np
            import cv2  # type: ignore

            arr = _np.frombuffer(img_bytes, dtype=_np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError("cv2.imdecode returned None")
            # OpenCV is BGR; convert to RGB
            im = im[:, :, ::-1].copy()
            return im
        except Exception as e:
            raise ValueError(f"Failed to decode image bytes to np array: {e}")

    def _np_to_pil(arr):
        from PIL import Image

        return Image.fromarray(arr, mode="RGB")

    def _np_to_tensor(arr):
        # Torch optional
        try:
            import torch  # type: ignore
            import numpy as _np

            if isinstance(arr, _np.ndarray):
                t = torch.from_numpy(arr)  # HWC, uint8
            else:
                t = torch.tensor(arr)
            # convert to CHW, float32 in [0,1]
            if t.ndim == 3 and t.shape[2] in (1, 3, 4):
                t = t.permute(2, 0, 1).contiguous()
            return t
        except Exception as e:
            raise ValueError(f"Tensor conversion unavailable: {e}")

    def _maybe_decode(img_bytes: bytes, h: int, w: int, c: int):
        if decode is None and images_return == "bytes":
            return img_bytes
        # always go through np for downstream conversions
        np_img = _decode_to_np(img_bytes, h, w, c)
        if images_return == "np" or decode == "np":
            return np_img
        if images_return == "pil" or decode == "pil":
            return _np_to_pil(np_img)
        if images_return == "tensor":
            return _np_to_tensor(np_img)
        return img_bytes

    # io for PIL
    import io  # used by PIL path

    # -------- materialize columns to Python lists --------
    if return_type == "arrow" and hasattr(table_like, "column_names"):

        def to_list(col):
            return _to_lists_arrow(table_like, col)

        n = table_like.num_rows

    elif return_type == "pandas" and hasattr(table_like, "columns"):

        def to_list(col):
            return _to_lists_pandas(table_like, col)

        n = len(table_like)

    elif isinstance(table_like, dict) and "columns" in table_like:
        # stub mode; synthesize empty lists of correct length if available
        n = int(table_like.get("rows", 0))

        def to_list(col):
            # stub: return [None]*n except for required cols where we return empty bytes
            if col == bytes_col:
                return [b""] * n
            return [None] * n
    else:
        raise ValueError(
            "Unsupported table_like; expected pyarrow.Table, pandas.DataFrame, or stub dict."
        )

    h_col, w_col, c_col = dims_cols

    bytes_list: List[bytes] = to_list(bytes_col)
    heights: List[Optional[int]] = to_list(h_col)
    widths: List[Optional[int]] = to_list(w_col)
    chans: List[Optional[int]] = to_list(c_col)

    labels: Optional[List[Any]] = (
        to_list(label_col)
        if (
            label_col
            and label_col
            in (
                getattr(table_like, "column_names", [])
                or getattr(table_like, "columns", [])
                or []
            )
        )
        else (None if label_col is None else to_list(label_col))
    )
    # If above branch failed due to stub, labels may be None or list of None.
    if label_col is None:
        labels = None

    if path_col is None:
        paths: List[Optional[str]] = [None] * n
    else:
        paths = to_list(path_col)

    # -------- iterate rows --------
    for i in range(n):
        b = bytes_list[i]
        h = int(heights[i]) if heights[i] is not None else None
        w = int(widths[i]) if widths[i] is not None else None
        c = int(chans[i]) if chans[i] is not None else None

        if not isinstance(b, (bytes, bytearray)):
            # pandas could store memoryview; Arrow guarantees bytes-like
            try:
                b = bytes(b) if b is not None else b""
            except Exception:
                raise ValueError(f"Row {i}: {bytes_col} is not bytes-like.")

        if h is None or w is None or c is None:
            print("WARNING h, w, or c is None")
            pass

        img_obj = _maybe_decode(b, h or 0, w or 0, c or 3)

        img_list = [img_obj]
        lbl_list = None if labels is None else [labels[i]]
        path_list = [paths[i] if i < len(paths) else None]

        yield (img_list, lbl_list, path_list)

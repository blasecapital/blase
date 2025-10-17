from typing import Any, Dict, Iterable, List
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from blase.utils.fs import ensure_dir
from blase.types import SinkResult, Artifact
from blase.preparing.interfaces import ManifestBatch, Splits


def _collect_rows(manifest_iter: Iterable[ManifestBatch]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for mb in manifest_iter:
        for r in mb.data:
            rows[r["image_id"]] = r
    return rows


def _row_view(r: Dict[str, Any]) -> Dict[str, Any]:
    # Convert lists of dicts to Arrow-friendly structures; keep as is for now using large_list+struct
    return {
        "image_id": r["image_id"],
        "path": r.get("path"),
        "sha256": r.get("sha256"),
        "height": r.get("height"),
        "width": r.get("width"),
        "det": r.get("det", []),  # list<struct<...>>
        "cls": r.get("cls", []),  # list<string|int>
    }


def _schema() -> pa.schema:
    det_struct = pa.struct(
        [
            ("xmin", pa.float32()),
            ("ymin", pa.float32()),
            ("xmax", pa.float32()),
            ("ymax", pa.float32()),
            ("label", pa.large_binary()),  # allow str or int via bytes
            ("iscrowd", pa.int32()),
        ]
    )
    return pa.schema(
        [
            ("image_id", pa.string()),
            ("path", pa.string()),
            ("sha256", pa.string()),
            ("height", pa.int32()),
            ("width", pa.int32()),
            ("det", pa.large_list(det_struct)),
            ("cls", pa.large_list(pa.large_binary())),
        ]
    )


def write(
    manifest_iter: Iterable[ManifestBatch],
    splits: Splits,
    cfg: Dict[str, Any],
) -> SinkResult:
    out_dir: Path = Path(cfg["out_dir"])
    ensure_dir(out_dir)
    order_key: str = cfg.get("order_key", "sha256")
    deterministic: bool = bool(cfg.get("deterministic_order", True))

    rows = _collect_rows(manifest_iter)
    artifacts: List[Artifact] = []

    for split_name, id_list in splits.items():
        ids = [iid for iid in id_list if iid in rows]
        if deterministic:
            ids.sort(key=lambda iid: str(rows[iid].get(order_key) or iid))

        data = [_row_view(rows[iid]) for iid in ids]

        # Build Arrow table with explicit schema
        sch = _schema()
        tbl = pa.Table.from_pylist(data, schema=sch)
        sidecar_path = out_dir / f"{split_name}.labels.parquet"
        pq.write_table(tbl, sidecar_path, compression="zstd", use_dictionary=True)
        artifacts.append(
            Artifact(path=str(sidecar_path), kind="sidecar_parquet", data_hash=None)
        )

    return SinkResult(
        is_last=True,
        meta={"format": "parquet", "order_key": order_key},
        artifacts=artifacts,
    )

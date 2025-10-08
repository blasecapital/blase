from typing import Any, Dict, Iterable, List
from pathlib import Path
import json

from blase.utils.fs import ensure_dir
from blase.types import SinkResult, Artifact
from blase.preparing.interfaces import ManifestBatch, Splits


def _collect_rows(manifest_iter: Iterable[ManifestBatch]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for mb in manifest_iter:
        for r in mb.data:
            rows[r["image_id"]] = r
    # synthesize meta from last batch if present
    return rows


def _row_view(r: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal, stable schema for sidecars
    return {
        "image_id": r["image_id"],
        "path": r.get("path"),
        "sha256": r.get("sha256"),
        "height": r.get("height"),
        "width": r.get("width"),
        "det": r.get("det", []),
        "cls": r.get("cls", []),
    }


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

        sidecar_path = out_dir / f"{split_name}.labels.jsonl"
        with sidecar_path.open("w", encoding="utf-8") as fh:
            for iid in ids:
                fh.write(json.dumps(_row_view(rows[iid])) + "\n")

        artifacts.append(
            Artifact(path=str(sidecar_path), kind="sidecar_jsonl", data_hash=None)
        )

    return SinkResult(
        is_last=True,
        meta={"format": "jsonl", "order_key": order_key},
        artifacts=artifacts,
    )

import os
import sys
import json
import sqlite3
import subprocess
import importlib
from types import ModuleType
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

# ---------- helpers ----------


def import_blase_fresh(project_root: Path) -> ModuleType:
    os.environ["BLASE_HOME"] = str(project_root)
    cwd = os.getcwd()
    os.chdir(project_root)
    try:
        for m in [
            m for m in list(sys.modules) if m == "blase" or m.startswith("blase.")
        ]:
            sys.modules.pop(m, None)
        return importlib.import_module("blase")
    finally:
        os.chdir(cwd)


def workspace(tmp_path, monkeypatch):
    project_root = tmp_path / "proj_sidecar_restore"
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "runs").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "working").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(project_root)
    monkeypatch.setenv("BLASE_HOME", str(project_root))
    return project_root


def _latest_run_path(project_root: Path) -> Path:
    runs = project_root / "runs"
    cands = []
    for d in runs.iterdir():
        if not d.is_dir():
            continue
        db = d / "nodes" / "nodes.db"
        if db.exists():
            cands.append((db.stat().st_mtime, d))
    assert cands, "no runs found"
    cands.sort(reverse=True)
    return cands[0][1]


def _con(project_root: Path) -> sqlite3.Connection:
    rp = _latest_run_path(project_root)
    db = rp / "nodes" / "nodes.db"
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def latest_step(project_root: Path, fqn: str) -> str:
    c = _con(project_root)
    try:
        row = c.execute(
            """
            SELECT step_hash
            FROM steps
            WHERE function_fqn=? AND status IN ('completed','complete','done','success','ok')
            ORDER BY ts_start DESC
            LIMIT 1
            """,
            (fqn,),
        ).fetchone()
        assert row, f"no completed {fqn} step recorded"
        return row["step_hash"]
    finally:
        c.close()


def outputs_for_step(project_root: Path, step_hash: str):
    c = _con(project_root)
    try:
        rows = c.execute(
            "SELECT data_hash, name FROM step_outputs WHERE step_hash=? ORDER BY name ASC",
            (step_hash,),
        ).fetchall()
        return [{"data_hash": r["data_hash"], "name": r["name"]} for r in rows]
    finally:
        c.close()


def cli_restore_step_to(project_root: Path, step_hash: str, to_dir: Path):
    run_path = _latest_run_path(project_root)
    cmd = [
        sys.executable,
        "-m",
        "blase.cli.cli",
        "restore",
        "run",
        "--run",
        str(run_path),
        "--step",
        step_hash,
        "--mode",
        "replay",
        "--to",
        str(to_dir),
        "--on-conflict",
        "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


def cli_restore_data_to(project_root: Path, data_hash: str, to_path: Path):
    run_path = _latest_run_path(project_root)
    cmd = [
        sys.executable,
        "-m",
        "blase.cli.cli",
        "restore",
        "run",
        "--run",
        str(run_path),
        "--data",
        data_hash,
        "--mode",
        "replay",
        "--to",
        str(to_path),
        "--on-conflict",
        "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


def write_parquet_images(path: Path):
    tbl = pa.table(
        {
            "image_id": pa.array(["a", "b", "c"]),
            "path": pa.array(["a.jpg", "b.jpg", "c.jpg"]),
            "height": pa.array([2, 4, 4], type=pa.int32()),
            "width": pa.array([4, 8, 8], type=pa.int32()),
            "sha256": pa.array(["aa", "bb", "cc"]),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path)


def write_tiny_coco(path: Path):
    data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 4, "height": 2},
            {"id": 2, "file_name": "b.jpg", "width": 8, "height": 4},
            {"id": 3, "file_name": "c.jpg", "width": 8, "height": 4},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "bbox": [0, 0, 2, 1],
                "category_id": 5,
                "iscrowd": 0,
            },
            {
                "id": 11,
                "image_id": 2,
                "bbox": [2, 1, 2, 2],
                "category_id": 6,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 5, "name": "weed"}, {"id": 6, "name": "radish"}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------- test ----------


@pytest.mark.integration
def test_write_label_sidecars_step_and_data_restore(tmp_path, monkeypatch):
    project_root = workspace(tmp_path, monkeypatch)
    pq_path = project_root / "data" / "working" / "images_000.parquet"
    coco_path = project_root / "data" / "working" / "labels.json"
    write_parquet_images(pq_path)
    write_tiny_coco(coco_path)

    blase = import_blase_fresh(project_root)
    Prepare = blase.Prepare
    ImageConfig = blase.prepare.ImageConfig
    JoinConfig = blase.prepare.JoinConfig
    BoxConfig = blase.prepare.BoxConfig
    ClassConfig = blase.prepare.ClassConfig
    ScaleConfig = blase.prepare.ScaleConfig
    images_parquet = blase.prepare.images_parquet
    coco = blase.prepare.coco

    prep = Prepare()

    # manifest (tracked)
    manifest = prep.build_manifest(
        data_sources=[images_parquet(str(pq_path))],
        label_sources=[coco(str(coco_path))],
        image_cfg=ImageConfig(bytes_col=None, sha256_col="sha256"),
        join_cfg=JoinConfig(image_id_resolver="provided", keep_unlabeled_images=True),
        box_cfg=BoxConfig(coord_in="xyxy_abs", coord_out="xyxy_rel", clamp_boxes=True),
        class_cfg=ClassConfig(),
        scale_cfg=ScaleConfig(join_strategy="duckdb_temp", batch_rows=2, seed=123),
        track=True,
    )

    # split (tracked)
    splits_b = prep.split(
        manifest=manifest,
        method="stratified",
        train=0.67,
        val=0.33,
        test=0.0,
        seed=123,
        track=True,
    )
    assert "train" in splits_b.data and "val" in splits_b.data

    # write sidecars (tracked)
    out_dir = project_root / "data" / "working" / "label_sidecar_orig"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = prep.write_label_sidecars(
        manifest=prep.build_manifest(
            data_sources=[images_parquet(str(pq_path))],
            label_sources=[coco(str(coco_path))],
            image_cfg=ImageConfig(bytes_col=None, sha256_col="sha256"),
            join_cfg=JoinConfig(
                image_id_resolver="provided", keep_unlabeled_images=True
            ),
            box_cfg=BoxConfig(
                coord_in="xyxy_abs", coord_out="xyxy_rel", clamp_boxes=True
            ),
            class_cfg=ClassConfig(),
            scale_cfg=ScaleConfig(join_strategy="duckdb_temp", batch_rows=2, seed=123),
            track=True,
        ),
        splits=splits_b.data,
        out_dir=out_dir,
        format="jsonl",
        deterministic_order=True,
        order_key="image_id",
        track=True,
    )
    artifacts = [Path(a.path) for a in (res.artifacts or [])]
    assert artifacts and all(p.exists() for p in artifacts)
    assert any(p.stat().st_size > 0 for p in artifacts)

    # grab recorded step + outputs
    step_hash = latest_step(project_root, "blase.Prepare.write_label_sidecars")
    outs = outputs_for_step(project_root, step_hash)
    assert outs, "write_label_sidecars step recorded no outputs"
    # prefer jsonl
    sidecar_out = next(
        (o for o in outs if (o["name"] or "").lower().endswith(".jsonl")), outs[0]
    )
    sidecar_hash = sidecar_out["data_hash"]

    # STEP RESTORE → new dir
    replay_dir = project_root / "data" / "working" / "label_sidecar_replay_step"
    if replay_dir.exists():
        for p in replay_dir.glob("*"):
            p.unlink()
    replay_dir.mkdir(parents=True, exist_ok=True)
    rc = cli_restore_step_to(project_root, step_hash, replay_dir)
    assert rc.returncode == 0, (
        f"step replay failed\nSTDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}"
    )
    # Handler returns list of produced paths; we just validate the expected files exist in --to dir.
    expected = [
        replay_dir / "train.labels.jsonl",
        replay_dir / "val.labels.jsonl",
        replay_dir / "test.labels.jsonl",
    ]
    # Depending on splits, some may be empty; assert at least one is produced and non-empty,
    # and that all expected paths (if present) are files.
    produced = [p for p in expected if p.exists()]
    assert produced, "step replay produced no sidecar files in the target dir"
    assert all(p.is_file() for p in produced)
    assert any(p.stat().st_size > 0 for p in produced)

    # DATA RESTORE → specific hash to exact file
    data_target = project_root / "data" / "working" / f"{sidecar_hash}.jsonl"
    if data_target.exists():
        data_target.unlink()
    rc2 = cli_restore_data_to(project_root, sidecar_hash, data_target)
    assert rc2.returncode == 0, (
        f"data restore failed\nSTDOUT:\n{rc2.stdout}\nSTDERR:\n{rc2.stderr}"
    )
    assert data_target.exists() and data_target.stat().st_size > 0

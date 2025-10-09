import os, sys, json, sqlite3, subprocess, importlib
from types import ModuleType
from pathlib import Path
import pytest
import pandas as pd
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
    project_root = tmp_path / "proj_compute_stats"
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "runs").mkdir(parents=True, exist_ok=True)
    data_root = project_root / "data" / "source_of_truth"
    data_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(project_root)
    monkeypatch.setenv("BLASE_HOME", str(project_root))
    return project_root, data_root


def _latest_run_path(project_root: Path) -> Path:
    runs = project_root / "runs"
    cands = []
    for d in runs.iterdir():
        if not d.is_dir():
            continue
        db = d / "nodes" / "nodes.db"
        if db.exists():
            mt = db.stat().st_mtime
            cands.append((mt, d))
    assert cands, "no runs found"
    cands.sort(reverse=True)
    return cands[0][1]


def _con(project_root: Path) -> sqlite3.Connection:
    rp = _latest_run_path(project_root)
    db = rp / "nodes" / "nodes.db"
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def latest_compute_stats_step(project_root: Path) -> str:
    c = _con(project_root)
    try:
        row = c.execute(
            """
            SELECT step_hash
            FROM steps
            WHERE function_fqn='blase.Prepare.compute_stats'
              AND status IN ('completed','complete','done','success','ok')
            ORDER BY ts_start DESC
            LIMIT 1
            """
        ).fetchone()
        assert row, "no completed Prepare.compute_stats step recorded"
        return row["step_hash"]
    finally:
        c.close()


def cli_restore_step(project_root: Path, step_hash: str):
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
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


def write_parquet_images(path: Path):
    df = pd.DataFrame(
        [
            {"image_id": "a", "path": "a.jpg", "height": 2, "width": 4, "sha256": "aa"},
            {"image_id": "b", "path": "b.jpg", "height": 4, "width": 8, "sha256": "bb"},
            {"image_id": "c", "path": "c.jpg", "height": 4, "width": 8, "sha256": "cc"},
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), path)


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
def test_compute_stats_then_replay(tmp_path, monkeypatch):
    project_root, data_root = workspace(tmp_path, monkeypatch)
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

    prep = Prepare()  # uses default_registry()

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

    # record compute_stats with tracking; this should capture manifest inputs
    stats_b = prep.compute_stats(manifest, by="class", track=True)
    assert isinstance(stats_b.data, dict)
    assert "class_hist" in stats_b.data

    # verify inputs were recorded on the compute_stats step
    step_hash = latest_compute_stats_step(project_root)
    c = _con(project_root)
    try:
        inputs = c.execute(
            "SELECT role FROM step_inputs WHERE step_hash=?", (step_hash,)
        ).fetchall()
        roles = {r["role"] for r in inputs}
        assert roles & {"manifest", "manifest_root"}, (
            "compute_stats missing manifest inputs"
        )
    finally:
        c.close()

    # replay compute_stats; handler should succeed with no side effects
    rc = cli_restore_step(project_root, step_hash)
    assert rc.returncode == 0, (
        f"replay failed\nSTDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}"
    )

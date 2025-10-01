import os
import sys
import sqlite3
import importlib
import subprocess
from types import ModuleType
from pathlib import Path

import pytest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# ---------------- Helpers ----------------


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
    project_root = tmp_path / "proj_preview_images"
    project_root.mkdir()
    monkeypatch.chdir(project_root)
    monkeypatch.setenv("BLASE_HOME", str(project_root))
    (project_root / "runs").mkdir(parents=True, exist_ok=True)
    data_root = project_root / "data" / "source_of_truth" / "flowers" / "jpg"
    data_root.mkdir(parents=True, exist_ok=True)
    return project_root, data_root


def gen_images(data_root: Path, n=6, size=(32, 24)):
    for i in range(n):
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[..., 0] = (i * 17) % 256
        arr[..., 1] = (i * 37) % 256
        arr[..., 2] = (i * 67) % 256
        img = Image.fromarray(arr, mode="RGB")
        name = f"flower_{i}.jpg"
        (data_root / name).parent.mkdir(parents=True, exist_ok=True)
        img.save(data_root / name, format="JPEG", quality=90)
    return data_root


def _latest_run_path(project_root: Path) -> Path:
    runs = project_root / "runs"
    assert runs.exists()
    cands = []
    for d in runs.iterdir():
        if not d.is_dir():
            continue
        db = d / "nodes" / "nodes.db"
        if db.exists():
            try:
                mt = db.stat().st_mtime
            except Exception:
                mt = d.stat().st_mtime
            cands.append((mt, d))
    assert cands, "No runs with nodes/nodes.db found"
    cands.sort(reverse=True)
    return cands[0][1]


def _con(project_root: Path) -> sqlite3.Connection:
    run_path = _latest_run_path(project_root)
    db = run_path / "nodes" / "nodes.db"
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def latest_preview_step(project_root: Path) -> str:
    c = _con(project_root)
    try:
        row = c.execute(
            """
            SELECT step_hash
            FROM steps
            WHERE function_fqn='blase.Examine.preview_images'
              AND status IN ('completed','complete','done','success','ok')
            ORDER BY ts_start DESC
            LIMIT 1
            """
        ).fetchone()
        assert row, "No completed Examine.preview_images step recorded"
        return row["step_hash"]
    finally:
        c.close()


def cli_restore_step(project_root: Path, step_hash: str, target_path: Path):
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
        str(target_path),
        "--on-conflict",
        "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


def make_parquet_with_bytes(path: Path, rows=3, size=(32, 32)):
    imgs = []
    heights, widths, chans = [], [], []
    for i in range(rows):
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[..., 0] = (i * 23) % 256
        arr[..., 1] = (i * 11) % 256
        arr[..., 2] = (i * 7) % 256
        im = Image.fromarray(arr, mode="RGB")
        import io

        buf = io.BytesIO()
        im.save(buf, format="PNG")
        imgs.append(buf.getvalue())
        heights.append(size[1])
        widths.append(size[0])
        chans.append(3)
    tbl = pa.table(
        {
            "img_bytes": pa.array(imgs, type=pa.binary()),
            "height": pa.array(heights, type=pa.int32()),
            "width": pa.array(widths, type=pa.int32()),
            "channels": pa.array(chans, type=pa.int32()),
            "label": pa.array(list(range(rows)), type=pa.int32()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path)


# ---------------- Pipelines ----------------


def run_images_to_parquet(project_root: Path, src_dir: Path, shard_prefix="flowers"):
    blase = import_blase_fresh(project_root)
    ex, ld = blase.Extract(), blase.Load()

    batches = ex.read_images(
        directory=str(src_dir),
        pattern="**/*.jpg",
        mode="manual",
        batch_size=3,
        return_type="np",
        track=True,
    )
    out_paths = []
    try:
        for batch in batches:
            res = ld.save_images_to_parquet(
                batch=batch,
                encode="jpeg",
                jpeg_quality=90,
                include_paths=False,  # exercise bytes-only path
                subdir="image_parquet_seed",
                shard_prefix=shard_prefix,
                compression="zstd",
                on_conflict="overwrite",
                track=True,
                use_blase_path=False,
                path="data/working",
            )
            out_paths.append(Path(res.artifacts[0].path))
    finally:
        try:
            batches.close()
        except Exception:
            pass
    return out_paths


# ---------------- Tests ----------------


@pytest.mark.integration
def test_preview_images_parquet_bytes_then_replay(tmp_path, monkeypatch):
    project_root, data_root = workspace(tmp_path, monkeypatch)
    gen_images(data_root, n=6)

    # Seed parquet shards
    shards = run_images_to_parquet(project_root, data_root)
    assert shards and all(p.exists() for p in shards)

    # Preview from one parquet (bytes mode), save individual thumbs
    blase = import_blase_fresh(project_root)
    exm = blase.Examine()
    thumbs_dir = project_root / "data" / "working" / "examine_test" / "thumbs_bytes"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    res = exm.preview_images(
        source=str(shards[0]),
        source_kind="parquet",
        parquet_image_bytes_col="img_bytes",
        sample_size=2,
        save_individual=True,
        out_dir=str(thumbs_dir),
        track=True,
    )
    # Validate stats and thumbs persisted
    assert res.meta["sample_size_got"] >= 1
    assert res.meta["ok_count"] >= 1
    written = [Path(p) for p in res.meta.get("written", [])]
    assert any(p.parent == thumbs_dir for p in written)

    # Replay the preview step to a grid path
    step_hash = latest_preview_step(project_root)
    grid_path = project_root / "data" / "working" / "examine_test" / "grid_replay.png"
    if grid_path.exists():
        grid_path.unlink()
    rc = cli_restore_step(project_root, step_hash, grid_path)
    assert rc.returncode == 0, (
        f"replay failed:\nSTDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}"
    )
    # Handler returns path; grid may be written if bytes-mode saver is used
    # Accept either a grid or only thumbs; ensure no error and directory still valid
    # If grid target was respected, file exists
    if grid_path.exists():
        assert grid_path.is_file()


@pytest.mark.integration
def test_preview_images_directory_mode(tmp_path, monkeypatch):
    project_root, data_root = workspace(tmp_path, monkeypatch)
    # Create source images
    gen_images(data_root, n=4)

    blase = import_blase_fresh(project_root)
    exm = blase.Examine()

    # Directory mode: persist grid + thumbs
    out_root = project_root / "data" / "working" / "examine_dir"
    grid_path = out_root / "preview.png"
    thumbs_dir = out_root / "thumbs"
    out_root.mkdir(parents=True, exist_ok=True)

    res = exm.preview_images(
        source=str(data_root),
        source_kind="directory",
        pattern="**/*.jpg",
        sample_size=3,
        save=True,
        to=str(grid_path),
        save_individual=True,
        out_dir=str(thumbs_dir),
        track=True,
    )

    assert res.meta["ok_count"] >= 1
    # Grid written
    assert grid_path.exists()
    # Some thumbs written
    thumbs = list(thumbs_dir.glob("thumb_*.png")) + list(thumbs_dir.glob("thumb_*.jpg"))
    assert len(thumbs) >= 1
    # Replay the preview step to a new grid path via CLI
    step_hash = latest_preview_step(project_root)
    grid_replay = out_root / "preview_replay.png"
    if grid_replay.exists():
        grid_replay.unlink()
    rc = cli_restore_step(project_root, step_hash, grid_replay)
    assert rc.returncode == 0, (
        f"replay failed:\nSTDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}"
    )
    assert grid_replay.exists()


@pytest.mark.integration
def test_preview_dir_nowrite_returns_meta(tmp_path, monkeypatch):
    project_root, data_root = workspace(tmp_path, monkeypatch)
    gen_images(data_root, n=4)

    blase = import_blase_fresh(project_root)
    exm = blase.Examine()

    res = exm.preview_images(
        source=str(data_root),
        source_kind="directory",
        pattern="**/*.jpg",
        sample_size=2,
        track=True,
    )
    assert isinstance(res.items, list)
    assert res.meta["source_kind"] == "directory"
    assert res.meta["ok_count"] >= 1
    assert res.meta.get("written", []) == []


@pytest.mark.integration
def test_preview_dir_grid_and_thumbs_materialize_and_replay_grid(tmp_path, monkeypatch):
    project_root, data_root = workspace(tmp_path, monkeypatch)
    gen_images(data_root, n=5)

    blase = import_blase_fresh(project_root)
    exm = blase.Examine()

    out_root = project_root / "data" / "working" / "examine_dir"
    grid_path = out_root / "preview.png"
    thumbs_dir = out_root / "thumbs"
    out_root.mkdir(parents=True, exist_ok=True)

    _ = exm.preview_images(
        source=str(data_root),
        source_kind="directory",
        pattern="**/*.jpg",
        sample_size=3,
        save=True,
        to=str(grid_path),
        save_individual=True,
        out_dir=str(thumbs_dir),
        track=True,
    )
    assert grid_path.exists()
    assert len(list(thumbs_dir.glob("thumb_*.png"))) >= 1

    step_hash = latest_preview_step(project_root)
    grid_replay = out_root / "preview_replay.png"
    if grid_replay.exists():
        grid_replay.unlink()
    rc = cli_restore_step(project_root, step_hash, grid_replay)
    assert rc.returncode == 0, f"replay failed:\n{rc.stdout}\n{rc.stderr}"
    assert grid_replay.exists()


@pytest.mark.integration
def test_preview_parquet_bytes_thumbs_materialize(tmp_path, monkeypatch):
    project_root, data_root = workspace(tmp_path, monkeypatch)
    pq_path = (
        project_root / "data" / "working" / "image_parquet" / "imgs-000001.parquet"
    )
    make_parquet_with_bytes(pq_path, rows=3)

    blase = import_blase_fresh(project_root)
    exm = blase.Examine()

    out_dir = project_root / "data" / "working" / "examine_parquet_bytes" / "thumbs"
    res = exm.preview_images(
        source=str(pq_path),
        source_kind="parquet",
        parquet_image_bytes_col="img_bytes",
        sample_size=2,
        save_individual=True,
        out_dir=str(out_dir),
        track=True,
    )
    thumbs = list(out_dir.glob("thumb_*.png")) + list(out_dir.glob("thumb_*.jpg"))
    assert len(thumbs) >= 1
    assert res.meta["ok_count"] >= 1
    assert res.meta["mode"] == "parquet_bytes"


@pytest.mark.integration
def test_preview_seed_determinism_and_clamp(tmp_path, monkeypatch):
    project_root, data_root = workspace(tmp_path, monkeypatch)
    gen_images(data_root, n=2)

    blase = import_blase_fresh(project_root)
    exm = blase.Examine()

    r1 = exm.preview_images(
        source=str(data_root),
        source_kind="directory",
        pattern="**/*.jpg",
        sample_size=10,  # clamp to population
        seed=123,
        track=True,
    )
    r2 = exm.preview_images(
        source=str(data_root),
        source_kind="directory",
        pattern="**/*.jpg",
        sample_size=10,
        seed=123,
        track=True,
    )
    assert r1.meta["sample_indices"] == r2.meta["sample_indices"]
    assert len(r1.items) <= 2


@pytest.mark.integration
def test_preview_replay_thumbs_when_missing(tmp_path, monkeypatch):
    project_root, data_root = workspace(tmp_path, monkeypatch)
    gen_images(data_root, n=3)

    blase = import_blase_fresh(project_root)
    exm = blase.Examine()

    out_root = project_root / "data" / "working" / "examine_dir2"
    thumbs_dir = out_root / "thumbs"
    out_root.mkdir(parents=True, exist_ok=True)

    _ = exm.preview_images(
        source=str(data_root),
        source_kind="directory",
        pattern="**/*.jpg",
        sample_size=2,
        save_individual=True,
        out_dir=str(thumbs_dir),
        track=True,
    )
    assert len(list(thumbs_dir.glob("thumb_*.*"))) >= 1

    # remove thumbs to simulate missing artifacts
    for p in thumbs_dir.glob("thumb_*.*"):
        p.unlink()
    assert len(list(thumbs_dir.glob("thumb_*.*"))) == 0

    # replay WITHOUT --to; handler should re-materialize thumbs
    step_hash = latest_preview_step(project_root)
    rc = cli_restore_step(project_root, step_hash, target_path=None)
    assert rc.returncode == 0, f"replay failed:\n{rc.stdout}\n{rc.stderr}"
    assert len(list(thumbs_dir.glob("thumb_*.*"))) >= 1

# tests/test_restore_image_parquet_from_parquet_pipeline.py

import os
import sys
import sqlite3
import importlib
import subprocess
from types import ModuleType
from pathlib import Path

import pytest
import numpy as np
from PIL import Image


# ---------------- Helpers ----------------


def import_blase_fresh(project_root: Path) -> ModuleType:
    os.environ["BLASE_HOME"] = str(project_root)
    os.chdir(project_root)
    for m in [m for m in list(sys.modules) if m == "blase" or m.startswith("blase.")]:
        sys.modules.pop(m, None)
    return importlib.import_module("blase")


def workspace(tmp_path, monkeypatch):
    project_root = tmp_path / "proj_img_parquet"
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
        name = f"flower_{i}0.jpg" if i % 2 == 0 else f"flower_{i}.jpg"
        (data_root / name).parent.mkdir(parents=True, exist_ok=True)
        img.save(data_root / name, format="JPEG", quality=90)
    return data_root


def _latest_run_path(project_root: Path) -> Path:
    roots = {project_root, project_root.parent}
    bh = Path(os.environ.get("BLASE_HOME", str(project_root)))
    roots.add(bh)
    roots.add(bh.parent)
    cands = []
    for root in roots:
        rr = root / "runs"
        if not rr.exists():
            continue
        for d in rr.iterdir():
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


def parquet_sink_step(project_root: Path) -> str:
    c = _con(project_root)
    try:
        row = c.execute(
            """
            SELECT step_hash
            FROM steps
            WHERE function_fqn='blase.Load.save_images_to_parquet'
              AND status IN ('completed','complete','done','success','ok')
            ORDER BY ts_start DESC LIMIT 1
            """
        ).fetchone()
        assert row, "No completed Load.save_images_to_parquet step recorded"
        return row["step_hash"]
    finally:
        c.close()


def parquet_data_hashes_for_step(project_root: Path, step_hash: str):
    c = _con(project_root)
    try:
        rows = c.execute(
            """
            SELECT name, data_hash
            FROM step_outputs
            WHERE step_hash=?
            ORDER BY CAST(substr(name, length('parquet_shard_') + 1) AS INTEGER) NULLS LAST
            """,
            (step_hash,),
        ).fetchall()
        return [r["data_hash"] for r in rows if r["name"].startswith("parquet_shard_")]
    finally:
        c.close()


def cli_restore_step(project_root: Path, step_hash: str, target_dir: Path):
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
        str(target_dir),
        "--on-conflict",
        "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


def cli_restore_data(project_root: Path, data_hash: str, target: Path):
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
        str(target),
        "--on-conflict",
        "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


def cli_materialize_data(project_root: Path, data_hash: str, target: Path):
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
        "materialize",
        "--to",
        str(target),
        "--on-conflict",
        "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


def run_images_to_parquet_once(
    project_root: Path, data_root: Path, shard_prefix="flowers"
):
    blase = import_blase_fresh(project_root)
    ex, tr, ld = blase.Extract(), blase.Transform(), blase.Load()

    def flip(batch):
        out = []
        for x in batch:
            if isinstance(x, Image.Image):
                out.append(x.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
            else:
                arr = np.asarray(x)
                out.append(arr[:, ::-1, ...])
        return out

    batches = ex.read_images(
        directory=str(data_root),
        pattern="**/*0.jpg",
        mode="manual",
        batch_size=2,
        return_type="np",
        track=True,
    )

    out_paths = []
    try:
        for batch in batches:
            batch = tr.apply_function(
                batch=batch,
                transform_func=flip,
                track=True,
            )
            res = ld.save_images_to_parquet(
                batch=batch,
                encode="jpeg",
                jpeg_quality=90,
                include_paths=True,
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


def run_parquet_to_parquet_once(
    project_root: Path, source_dir: Path, shard_prefix="flowers_r"
):
    blase = import_blase_fresh(project_root)
    ex, tr, ld = blase.Extract(), blase.Transform(), blase.Load()

    def flip(batch):
        out = []
        for x in batch:
            if isinstance(x, Image.Image):
                out.append(x.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
            else:
                arr = np.asarray(x)
                out.append(arr[:, ::-1, ...])
        return out

    batches = ex.read_parquet(
        source=str(source_dir),
        mode="images",
        return_type="arrow",
        decode="np",
        images_return="np",
        batch_size=2,
        track=True,
    )

    out_paths = []
    try:
        for batch in batches:
            batch = tr.apply_function(
                batch=batch,
                transform_func=flip,
                track=True,
            )
            res = ld.save_images_to_parquet(
                batch=batch,
                encode="jpeg",
                jpeg_quality=90,
                include_paths=True,
                subdir="image_parquet_from_parquet",
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
def test_restore_cli_parquet_read_then_materialize_and_replay(tmp_path, monkeypatch):
    from blase.utils.hashing import Hash

    project_root, data_root = workspace(tmp_path, monkeypatch)
    gen_images(data_root, n=6)

    # Seed: images -> parquet
    seed_paths = run_images_to_parquet_once(
        project_root, data_root, shard_prefix="flowers"
    )
    assert seed_paths and all(p.exists() for p in seed_paths)
    seed_dir = seed_paths[0].parent  # data/working/image_parquet_seed

    # Read parquet -> transform -> parquet
    out_paths = run_parquet_to_parquet_once(
        project_root, seed_dir, shard_prefix="flowers_r"
    )
    assert out_paths and all(p.exists() for p in out_paths)

    # Identify latest parquet sink step and its data hashes
    shash = parquet_sink_step(project_root)
    shard_hashes = parquet_data_hashes_for_step(project_root, shash)
    assert shard_hashes, "No parquet shard hashes recorded"
    assert len(shard_hashes) == len(out_paths)

    # Delete produced shards to force restore
    for p in out_paths:
        p.unlink()
        assert not p.exists()

    # Step replay
    step_target_dir = project_root / "restore_step_parquet_from_parquet"
    step_target_dir.mkdir(parents=True, exist_ok=True)
    res = cli_restore_step(project_root, shash, step_target_dir)
    assert res.returncode == 0, (
        f"Step replay failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
    step_outs = sorted(step_target_dir.glob("*.parquet"))
    assert len(step_outs) >= len(shard_hashes)

    # Data materialize
    mat_dir = project_root / "restore_mat_parquet_from_parquet"
    mat_dir.mkdir(parents=True, exist_ok=True)
    for i, dh in enumerate(shard_hashes, 1):
        target = mat_dir / f"mat_{i}.parquet"
        if target.exists():
            target.unlink()
        res = cli_materialize_data(project_root, dh, target)
        assert res.returncode == 0, (
            f"Materialize failed for shard {i}:\n{res.stdout}\n{res.stderr}"
        )
        assert target.exists()
        assert Hash().hash_file(target) == dh

    # Data replay
    rep_dir = project_root / "restore_replay_parquet_from_parquet"
    rep_dir.mkdir(parents=True, exist_ok=True)
    for i, dh in enumerate(shard_hashes, 1):
        target = rep_dir / f"rep_{i}.parquet"
        if target.exists():
            target.unlink()
        res = cli_restore_data(project_root, dh, target)
        assert res.returncode == 0, (
            f"Data replay failed for shard {i}:\n{res.stdout}\n{res.stderr}"
        )
        assert target.exists()
        assert Hash().hash_file(target) == dh

# tests/test_restore_image_parquet_pipeline.py

import os
import sys
import csv
import json
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
    # purge any cached blase modules
    for m in [m for m in list(sys.modules) if m == "blase" or m.startswith("blase.")]:
        sys.modules.pop(m, None)
    return importlib.import_module("blase")


def workspace(tmp_path, monkeypatch):
    project_root = tmp_path / "proj_img"
    project_root.mkdir()
    monkeypatch.chdir(project_root)
    monkeypatch.setenv("BLASE_HOME", str(project_root))
    (project_root / "runs").mkdir(parents=True, exist_ok=True)
    data_root = project_root / "data" / "source_of_truth" / "flowers" / "jpg"
    data_root.mkdir(parents=True, exist_ok=True)
    return project_root, data_root


def gen_images(data_root: Path, n=6, size=(32, 24)):
    # create small JPEGs with deterministic pixels
    for i in range(n):
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[..., 0] = (i * 17) % 256
        arr[..., 1] = (i * 37) % 256
        arr[..., 2] = (i * 67) % 256
        img = Image.fromarray(arr, mode="RGB")
        # ensure some files end with ...0.jpg to match the pattern in example
        name = f"flower_{i}0.jpg" if i % 2 == 0 else f"flower_{i}.jpg"
        (data_root / name).parent.mkdir(parents=True, exist_ok=True)
        img.save(data_root / name, format="JPEG", quality=90)
    return data_root


def _latest_run_path(project_root: Path) -> Path:
    roots = {project_root, project_root.parent}
    bh = Path(os.environ.get("BLASE_HOME", str(project_root)))
    roots.add(bh); roots.add(bh.parent)
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
        row = c.execute("""
            SELECT step_hash
            FROM steps
            WHERE function_fqn='blase.Load.save_images_to_parquet'
              AND status IN ('completed','complete','done','success','ok')
            ORDER BY ts_start DESC LIMIT 1
        """).fetchone()
        assert row, "No completed Load.save_images_to_parquet step recorded"
        return row["step_hash"]
    finally:
        c.close()


def parquet_data_hashes_for_step(project_root: Path, step_hash: str):
    c = _con(project_root)
    try:
        rows = c.execute("""
            SELECT name, data_hash
            FROM step_outputs
            WHERE step_hash=?
            ORDER BY CAST(substr(name, length('parquet_shard_') + 1) AS INTEGER) NULLS LAST
        """, (step_hash,)).fetchall()
        # keep only parquet_shard_* rows
        return [r["data_hash"] for r in rows if r["name"].startswith("parquet_shard_")]
    finally:
        c.close()


def run_image_pipeline_once(project_root: Path, data_root: Path, *, track=True, shard_prefix="flowers"):
    blase = import_blase_fresh(project_root)
    ex, tr, ld = blase.Extract(), blase.Transform(), blase.Load()

    def modify_images(batch):
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
        track=track,
    )

    out_paths = []
    try:
        for paths, is_last, images, meta in batches:
            out, last, meta = tr.apply_function(
                data=images,
                transform_func=modify_images,
                last_batch=is_last,
                meta=meta,
                track=track,
            )
            shard_path, last_batch, _meta = ld.save_images_to_parquet(
                data=(out, None, paths),
                last_batch=last,
                meta=meta,
                encode="jpeg",
                jpeg_quality=90,
                include_paths=True,
                subdir="image_parquet",
                shard_prefix=shard_prefix,
                compression="zstd",
                on_conflict="overwrite",
                track=track,
                use_blase_path=False,
                path="data/working",
            )
            out_paths.append(Path(shard_path))
    finally:
        try:
            batches.close()
        except Exception:
            pass
    return out_paths


def cli_restore_step(project_root: Path, step_hash: str, target_dir: Path):
    run_path = _latest_run_path(project_root)
    cmd = [
        sys.executable, "-m", "blase.cli.cli",
        "restore", "run",
        "--run", str(run_path),
        "--step", step_hash,
        "--mode", "replay",
        "--to", str(target_dir),
        "--on-conflict", "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


def cli_restore_data(project_root: Path, data_hash: str, target_dir: Path):
    run_path = _latest_run_path(project_root)
    cmd = [
        sys.executable, "-m", "blase.cli.cli",
        "restore", "run",
        "--run", str(run_path),
        "--data", data_hash,
        "--mode", "replay",
        "--to", str(target_dir),
        "--on-conflict", "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


def cli_materialize_data(project_root: Path, data_hash: str, target_dir: Path):
    run_path = _latest_run_path(project_root)
    cmd = [
        sys.executable, "-m", "blase.cli.cli",
        "restore", "run",
        "--run", str(run_path),
        "--data", data_hash,
        "--mode", "materialize",
        "--to", str(target_dir),
        "--on-conflict", "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


# ---------------- Tests ----------------

@pytest.mark.integration
def test_restore_cli_image_parquet_materialize_and_replay(tmp_path, monkeypatch):
    from blase.utils.hashing import Hash

    project_root, data_root = workspace(tmp_path, monkeypatch)
    gen_images(data_root, n=6)

    # Run image pipeline once → produce shards
    out_paths = run_image_pipeline_once(project_root, data_root, track=True, shard_prefix="flowers")
    assert out_paths, "No parquet shards produced"
    for p in out_paths:
        assert p.exists()

    # Gather sink step and data hashes for shards
    shash = parquet_sink_step(project_root)
    shard_hashes = parquet_data_hashes_for_step(project_root, shash)
    assert len(shard_hashes) == len(out_paths)

    # Hash each shard then delete to force restore
    shard_hashes_fp = [(Hash().hash_file(p), p.name) for p in out_paths]
    for p in out_paths:
        p.unlink()
        assert not p.exists()

    # ---- Step replay: write shards into a target directory
    step_target_dir = project_root / "restore_step_parquet"
    step_target_dir.mkdir(parents=True, exist_ok=True)
    res = cli_restore_step(project_root, shash, step_target_dir)
    assert res.returncode == 0, f"CLI step replay failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Expect shards present again (names may be deterministic per handler)
    step_outs = sorted(step_target_dir.glob("*.parquet"))
    assert len(step_outs) >= len(shard_hashes), "Expected at least as many shards after step replay"

    # ---- Data materialize: each shard by its data hash
    mat_dir = project_root / "restore_mat_parquet"
    mat_dir.mkdir(parents=True, exist_ok=True)
    for i, dh in enumerate(shard_hashes, 1):
        target = mat_dir / f"mat_{i}.parquet"
        if target.exists():
            target.unlink()
        res = cli_materialize_data(project_root, dh, target)
        assert res.returncode == 0, f"CLI materialize failed for shard {i}:\n{res.stdout}\n{res.stderr}"
        assert target.exists()
        # materialize path should match bytes
        assert Hash().hash_file(target) == dh

    # ---- Data replay: regenerate shards by data hash
    rep_dir = project_root / "restore_replay_parquet"
    rep_dir.mkdir(parents=True, exist_ok=True)
    for i, dh in enumerate(shard_hashes, 1):
        target = rep_dir / f"rep_{i}.parquet"
        if target.exists():
            target.unlink()
        res = cli_restore_data(project_root, dh, target)
        assert res.returncode == 0, f"CLI data replay failed for shard {i}:\n{res.stdout}\n{res.stderr}"
        assert target.exists()
        # strict bytes check
        assert Hash().hash_file(target) == dh
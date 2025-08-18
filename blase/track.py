from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple
from pathlib import Path
import json
import time
import platform
import sys
import subprocess
import os
from datetime import datetime

from blase.tracking.step_backend import ensure_schema, StepContext, StreamStep
import blase.utils.config as config

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _write_active(run_root: Path, run_id: str, status: str = "active") -> None:
    (run_root / "active_run.blase").write_text(json.dumps({"run_id": run_id, "status": status}, indent=2))

class Track:
    """
    Single public API for run lifecycle + step/stream creation.
    Backed by tracking.step_backend.{StepContext,StreamStep}.
    """
    _singleton: Optional["Track"] = None

    def __init__(self, runs_dir: Optional[Path] = None, reuse: bool = True):
        self.runs_dir = Path(runs_dir or (Path.cwd() / "runs"))
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.run_root = self.runs_dir                  # <-- define run_root
        self._active_meta = self._load_or_create_active_run(reuse=reuse)
        self.run_id = self._active_meta["run_id"]
        self.run_path = self.runs_dir / self.run_id
        (self.run_path / "cas").mkdir(parents=True, exist_ok=True)
        (self.run_path / "nodes").mkdir(parents=True, exist_ok=True)
        ensure_schema(self.run_path)

    # ---------- class helpers ----------
    @classmethod
    def get(cls, enable: bool, *, runs_dir: Optional[Path] = None, reuse: bool = True) -> Optional["Track"]:
        """Preferred way to obtain a tracker from library code."""
        if not enable:
            return None
        if cls._singleton is None:
            cls._singleton = cls(runs_dir=runs_dir, reuse=reuse)
            cls._singleton.start_run()                 # ensure metadata & pointer
        return cls._singleton

    # ---------- public step APIs ----------
    def step(self, function_fqn: str, params: Dict[str, Any]) -> StepContext:
        return StepContext(self.run_path, function_fqn=function_fqn, params=params, run_id=self.run_id)

    def stream(self, function_fqn: str, params: Dict[str, Any], *, code_fn=None) -> StreamStep:
        return StreamStep(self, function_fqn, params, code_fn=code_fn)

    # ---------- run lifecycle ----------
    def start_run(self, run_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Idempotent: ensures schema + run.json exist; updates label/metadata if supplied."""
        ensure_schema(self.run_path)
        run_meta_path = self.run_path / "run.json"
        meta: Dict[str, Any] = {}
        if run_meta_path.exists():
            try:
                meta = json.loads(run_meta_path.read_text()) or {}
            except Exception:
                meta = {}
        meta.setdefault("run_id", self.run_id)
        meta.setdefault("created_at", _now())
        if run_name is not None:
            meta["label"] = run_name
        if metadata:
            meta.update(metadata)
        run_meta_path.write_text(json.dumps(meta, indent=2))
        # keep pointer fresh
        _write_active(self.run_root, self.run_id, status="active")   # <-- use self.run_root

    def end_run(self, status: str = "completed") -> None:
        active = self.runs_dir / "active_run.blase"
        if active.exists():
            meta = json.loads(active.read_text())
            meta["status"] = status
            meta["ended_at"] = _now()
            active.write_text(json.dumps(meta, indent=2))

    # ---------- properties ----------
    @property
    def paths(self):
        return {
            "project_root": config.PROJECT_ROOT,
            "data_sot":     config.DATA_SOURCE_OF_TRUTH,
            "data_working": config.DATA_WORKING,
            "restore_dir":  config.RESTORE_DEFAULT_DIR,
        }
    @property
    def cas_policy(self) -> str:
        return config.CAS_POLICY

    # ---------- internals ----------
    def _load_or_create_active_run(self, *, reuse: bool) -> Dict[str, Any]:
        active = self.runs_dir / "active_run.blase"
        if active.exists() and reuse:
            return json.loads(active.read_text())

        run_id = time.strftime("%Y-%m-%dT%H-%M-%S")
        run_path = self.runs_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        (run_path / "run.json").write_text(json.dumps({
            "created_at": _now(),
            "system": platform.system(),
            "platform": platform.platform(),
            "python_version": sys.version,
        }, indent=2))

        try:
            (run_path / "dependencies.txt").write_text(
                subprocess.check_output([sys.executable, "-m", "pip", "freeze"], timeout=15).decode("utf-8")
            )
        except Exception:
            pass

        meta = {"run_id": run_id, "status": "active", "created_at": _now()}
        active.write_text(json.dumps(meta, indent=2))
        return meta

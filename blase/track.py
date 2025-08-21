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
    Run lifecycle manager and single entry point for tracking steps/streams.

    ``Track`` encapsulates creation and management of a *run* directory
    (``runs/<run_id>``), including initialization of the content-addressable
    store (``cas/``) and the tracking database (``nodes/nodes.db``). It exposes
    public APIs to start/end runs and to create recorded step contexts for both
    single-call steps and streaming (batch) steps.

    The class is **intended as the public API** for tracking and is commonly
    used internally by higher-level objects (e.g., ``Extract``, ``Transform``,
    ``Load``) to record lineage automatically.

    Parameters
    ----------
    runs_dir : Path or None, optional
        Directory that contains all run subdirectories. If ``None``, defaults
        to ``<cwd>/runs``. The directory (and nested structure) is created
        on first use.
    reuse : bool, optional
        If ``True`` (default), reuse an existing ``active_run.blase`` pointer
        when present; otherwise create a new run and update the pointer.

    Attributes
    ----------
    runs_dir : Path
        Root directory for all runs (contains subdirectories per ``run_id``).
    run_root : Path
        Alias of ``runs_dir`` for convenience.
    run_id : str
        The identifier of the active run (e.g., timestamp-based).
    run_path : Path
        Filesystem path to the active run: ``runs_dir / run_id``.
    _active_meta : dict
        Metadata loaded/stored with the active run pointer.

    Notes
    -----
    - On initialization, ``Track`` ensures the following structure exists:
      ``runs/<run_id>/cas/`` and ``runs/<run_id>/nodes/``, then calls
      :func:`ensure_schema` to initialize the tracking DB schema.
    - A singleton instance is optionally maintained via :meth:`Track.get` to
      avoid repeated setup in library code. ``Track.get(enable=False)``
      returns ``None`` for code paths where tracking is disabled.
    - ``Track.step`` creates a *non-streaming* step context (single call),
      whereas ``Track.stream`` creates a *streaming* step that records batches.

    Examples
    --------
    Create a tracker and record a single step::

        trk = Track()               # creates runs/<run_id>/...
        trk.start_run("etl-run")    # optional label

        with trk.step("blase.Extract.read_csv", params={"file_path": "data.csv"}) as s:
            # ... your work ...
            s.record_output(data_hash=..., kind="csv", name="raw.csv")

        trk.end_run()

    Create a streaming step and record batches::

        trk = Track()
        ss = trk.stream("blase.Transform.apply_function", params={"fn": "my_fn"})
        for batch, last in upstream:
            with ss.next_batch(last=last) as sb:
                out = transform(batch)
                sb.record_output_batch(out_hash=..., kind="csv")
        trk.end_run()

    See Also
    --------
    StepContext
        Context manager used for non-streaming step recording.
    StreamStep
        Streaming/batched step helper used by :meth:`Track.stream`.
    """
    _singleton: Optional["Track"] = None

    def __init__(self, runs_dir: Optional[Path] = None, reuse: bool = True):
        self.runs_dir = Path(runs_dir or (Path.cwd() / "runs"))
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.run_root = self.runs_dir
        self._active_meta = self._load_or_create_active_run(reuse=reuse)
        self.run_id = self._active_meta["run_id"]
        self.run_path = self.runs_dir / self.run_id
        (self.run_path / "cas").mkdir(parents=True, exist_ok=True)
        (self.run_path / "nodes").mkdir(parents=True, exist_ok=True)
        ensure_schema(self.run_path)

    # ---------- class helpers ----------
    @classmethod
    def get(cls, enable: bool, *, runs_dir: Optional[Path] = None, reuse: bool = True) -> Optional["Track"]:
        """
        Return a process-wide tracker singleton if ``enable`` is True.

        Parameters
        ----------
        enable : bool
            If ``False``, return ``None`` (tracking disabled). If ``True``,
            return a singleton ``Track`` instance, creating it if necessary.
        runs_dir : Path or None, optional
            Root directory for runs. Passed through to the constructor on first use.
        reuse : bool, optional
            Whether to reuse an existing active run pointer when creating.

        Returns
        -------
        Track or None
            The singleton tracker or ``None`` if tracking is disabled.
        """
        if not enable:
            return None
        if cls._singleton is None:
            cls._singleton = cls(runs_dir=runs_dir, reuse=reuse)
            cls._singleton.start_run()                 # ensure metadata & pointer
        return cls._singleton

    # ---------- public step APIs ----------
    def step(self, function_fqn: str, params: Dict[str, Any]) -> StepContext:
        """
        Create a non-streaming step context for a single recorded function call.

        Parameters
        ----------
        function_fqn : str
            Fully-qualified function name (FQN) recorded for this step.
        params : dict
            Parameter dictionary persisted with the step metadata.

        Returns
        -------
        StepContext
            Context manager for recording inputs/outputs and status.

        Notes
        -----
        The returned :class:`StepContext` exposes :class:`StepOps` via its
        ``ops`` attribute for recording inputs/outputs.
        """
        return StepContext(self.run_path, function_fqn=function_fqn, params=params, run_id=self.run_id)

    def stream(self, function_fqn: str, params: Dict[str, Any], *, code_fn=None) -> StreamStep:
        """
        Create a streaming (batched) step helper.

        Parameters
        ----------
        function_fqn : str
            Fully-qualified function name (FQN) for the streaming step.
        params : dict
            Parameter dictionary persisted with the step metadata.
        code_fn : callable or None, keyword-only
            Optional function object used to capture code lineage.

        Returns
        -------
        StreamStep
            Helper for recording multiple batches under a single step.

        Notes
        -----
        :class:`StreamStep` internally manages a :class:`StepContext` and uses
        :class:`StepOps` to record lineage and status across batches.
        """
        return StreamStep(self, function_fqn, params, code_fn=code_fn)

    # ---------- run lifecycle ----------
    def start_run(self, run_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Idempotently initialize or update the active run metadata.

        Ensures the schema exists, creates/updates ``run.json``, and refreshes
        the ``active_run.blase`` pointer. If a label or extra metadata is
        provided, it is merged into the current run metadata.

        Parameters
        ----------
        run_name : str or None, optional
            Optional human-readable label for the run.
        metadata : dict or None, optional
            Additional metadata to merge into ``run.json``.

        Returns
        -------
        None
        """
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
        """
        Mark the active run as finished and persist end metadata.

        Parameters
        ----------
        status : str, optional
            Final run status to record (e.g., ``"completed"``, ``"failed"``).
            Defaults to ``"completed"``.

        Returns
        -------
        None
        """
        active = self.runs_dir / "active_run.blase"
        if active.exists():
            meta = json.loads(active.read_text())
            meta["status"] = status
            meta["ended_at"] = _now()
            active.write_text(json.dumps(meta, indent=2))

    # ---------- properties ----------
    @property
    def paths(self):
        """
        Common project paths used by the tracker.

        Returns
        -------
        dict
            Mapping with keys: ``project_root``, ``data_sot``, ``data_working``,
            ``restore_dir``.
        """
        return {
            "project_root": config.PROJECT_ROOT,
            "data_sot":     config.DATA_SOURCE_OF_TRUTH,
            "data_working": config.DATA_WORKING,
            "restore_dir":  config.RESTORE_DEFAULT_DIR,
        }
    @property
    def cas_policy(self) -> str:
        """
        Current CAS materialization policy.

        Returns
        -------
        str
            Policy name (e.g., ``"reuse"``).
        """
        return config.CAS_POLICY

    # ---------- internals ----------
    def _load_or_create_active_run(self, *, reuse: bool) -> Dict[str, Any]:
        """
        Load the active run pointer, or create a new run if absent.

        Parameters
        ----------
        reuse : bool
            If ``True``, use an existing ``active_run.blase`` when present.
            Otherwise, create a new run and update the pointer.

        Returns
        -------
        dict
            Active run metadata containing at least ``run_id`` and ``status``.
        """
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

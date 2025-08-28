import uuid
from typing import Dict, Optional, Any, Tuple, List
import contextlib
import json
import sqlite3
import time
from pathlib import Path
import shutil
import os

from blase.utils.hashing import Hash
from blase.tracking.snapshot import build_code_blob, build_env_manifest
from blase.utils.config import cas_policy as _cas_policy_default

# --- CAS and DAG local storage helpers ---

DDL = [
    # ----------------------------------------------------------------------
    # runs
    # Holds high-level runs; useful for grouping steps and audit trails.
    # ----------------------------------------------------------------------
    """CREATE TABLE IF NOT EXISTS runs (
         run_id        TEXT PRIMARY KEY,
         created_at    TEXT NOT NULL,
         label         TEXT,
         metadata_json TEXT
       )""",
    # ----------------------------------------------------------------------
    # steps (DAG nodes)
    # Each executed function call is a step. Steps reference inputs/outputs in CAS.
    # For restore, we walk from a target output back to a materialized checkpoint.
    # ----------------------------------------------------------------------
    """CREATE TABLE IF NOT EXISTS steps (
         step_hash     TEXT PRIMARY KEY,
         run_id        TEXT NOT NULL,
         step_id       TEXT,
         function_fqn  TEXT NOT NULL,
         params_json   TEXT NOT NULL,
         seeds_json    TEXT,
         ts_start      TEXT NOT NULL,
         ts_end        TEXT,
         attempt_id    TEXT,
         sig_hash      TEXT,
         status        TEXT NOT NULL CHECK (status IN ('running','completed','failed','aborted'))
       )""",
    # ----------------------------------------------------------------------
    # data (CAS)
    # Logical, content-addressed "blobs" (SMALL descriptors only in this design).
    # Examples:
    #   - code.blob:v1            (stored source snapshot)
    #   - image.file.meta:v1      (header-only per-image JSON: width/height/mode/bytes/corrupt)
    #   - image.manifest:v1       (manifest descriptor: scan params + stats)
    #   - image.batch.meta:v1     (batch descriptor: membership/limits/seeds)
    #   - dataset.checkpoint:v1   (descriptor for a materialized dataset file/dir)
    # The ACTUAL BYTES for checkpoints live on disk and are linked via materializations.
    # ----------------------------------------------------------------------
    """CREATE TABLE IF NOT EXISTS data (
         data_hash     TEXT PRIMARY KEY,
         kind          TEXT NOT NULL,
         version       TEXT NOT NULL,
         byte_len      INTEGER,
         source_path   TEXT,
         metadata_json TEXT
       )""",

    # ----------------------------------------------------------------------
    # step_inputs / step_outputs (edges in the compute DAG)
    # Restore uses these to plan which steps to re-run after jumping to a checkpoint.
    # ----------------------------------------------------------------------
    """CREATE TABLE IF NOT EXISTS step_inputs (
         step_hash TEXT NOT NULL,
         data_hash TEXT NOT NULL,
         role      TEXT NOT NULL, 
         arg_name  TEXT,
         PRIMARY KEY (step_hash, data_hash, role)
       )""",

    """CREATE TABLE IF NOT EXISTS step_outputs (
         step_hash TEXT NOT NULL,
         data_hash TEXT NOT NULL,
         name      TEXT NOT NULL,
         PRIMARY KEY (step_hash, name)
       )""",
    # ----------------------------------------------------------------------
    # datasets (MANIFESTS)
    # Make datasets first-class "manifests". One row per manifest/batch-like collection.
    # Used heavily by Inspect/CLI for integrity summaries and by restore as a stable anchor.
    # - 'kind' distinguishes manifests (image.manifest:v1), batches (image.batch.meta:v1), etc.
    # - 'manifest_hash' optionally points at a CAS descriptor in 'data' (e.g., a saved parquet manifest).
    #   If you also materialize to disk, link that via 'materializations'.
    # ----------------------------------------------------------------------
    """CREATE TABLE IF NOT EXISTS datasets (
         dataset_id     TEXT PRIMARY KEY,
         name           TEXT,
         kind           TEXT NOT NULL DEFAULT 'generic',
         created_at     TEXT,
         manifest_hash  TEXT,
         metadata_json  TEXT
       )""",
    # ----------------------------------------------------------------------
    # dataset_members (membership list with ORDER)
    # Ordered membership enables deterministic batching and reproducible replays.
    # We optionally cache a few fields (is_corrupt/width/height) to power fast CLI
    # "health checks" without JSON parsing over thousands of rows. These are hints
    # copied from data.metadata_json (image.file.meta:v1) at scan time.
    # ----------------------------------------------------------------------
    """CREATE TABLE IF NOT EXISTS dataset_members (
         dataset_id   TEXT NOT NULL,
         data_hash    TEXT NOT NULL,
         ordinal      INTEGER NOT NULL,
         role         TEXT DEFAULT 'item',
         is_corrupt   INTEGER,
         width        INTEGER,
         height       INTEGER,
         PRIMARY KEY (dataset_id, ordinal),
         UNIQUE (dataset_id, data_hash)
       )""",
    # ----------------------------------------------------------------------
    # materializations
    # Map logical CAS data_hash to on-disk path(s). This is THE checkpoint anchor
    # for restore: we jump to the newest suitable materialization upstream,
    # then replay steps forward using code blobs + params.
    # ----------------------------------------------------------------------
    """CREATE TABLE IF NOT EXISTS materializations (
         data_hash TEXT NOT NULL,
         path      TEXT NOT NULL,
         ts        TEXT NOT NULL,
         PRIMARY KEY (data_hash, path)
       )""",
    # ----------------------------------------------------------------------
    # Indexes (query speed for planners, inspectors, and CLIs)
    # ----------------------------------------------------------------------
    "CREATE INDEX IF NOT EXISTS idx_inputs_data           ON step_inputs(data_hash)",
    "CREATE INDEX IF NOT EXISTS idx_outputs_data          ON step_outputs(data_hash)",
    "CREATE INDEX IF NOT EXISTS idx_outputs_step          ON step_outputs(step_hash)",
    "CREATE INDEX IF NOT EXISTS idx_steps_sig             ON steps(sig_hash)",
    "CREATE INDEX IF NOT EXISTS idx_data_kind             ON data(kind)",

    # Helpful for CLI/Inspect: quick scan of all members of a dataset,
    # and reverse lookups from an item to its containing datasets.
    "CREATE INDEX IF NOT EXISTS idx_dataset_members_ds    ON dataset_members(dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_dataset_members_data  ON dataset_members(data_hash)",

    # Quickly filter manifests/batches by kind (images vs generic).
    "CREATE INDEX IF NOT EXISTS idx_datasets_kind         ON datasets(kind)",

    # This accelerates the lookup via datasets.manifest_hash.
    "CREATE INDEX IF NOT EXISTS idx_materializations_hash ON materializations(data_hash)"
]

def _conn(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(db)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("PRAGMA foreign_keys=OFF;")  # no FKs until you add migrations
    return c

def ensure_schema(run_path: Path) -> None:
    db = run_path / "nodes" / "nodes.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with _conn(db) as c:
        for stmt in DDL:
            c.execute(stmt)
        c.commit()

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def _normalize_params(p: Dict[str, Any]) -> Dict[str, Any]:
    # Strip non-serializables like Track objects, Path → str, etc.
    out = {}
    for k, v in p.items():
        if k in ("track", "_track", "logger"):
            continue
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            try:
                json.dumps(v)
                out[k] = v
            except Exception:
                out[k] = repr(v)
    return out

# --- CAS ---

class CAS:
    @staticmethod
    def put_file(path: Path, kind: str) -> Tuple[str, int]:
        hasher = Hash()
        file_hash = hasher.hash_file(path)
        return file_hash, os.path.getsize(path)
    @staticmethod
    def put_bytes(b: bytes, kind: str) -> Tuple[str, int]:
        hasher = Hash()
        h = hasher.hash_bytes(b)
        return h, len(b)
    
# --- hashing ---

def _attempt_id() -> str:
    return uuid.uuid4().hex

def canonical_signature_payload(
    *,
    function_fqn: str,
    params: Dict[str, Any],
    inputs: List[Tuple[str, str]],
    code_hash: Optional[str],
    env_hash: Optional[str],
    mats: Optional[Dict[str, Any]],
    seeds: Optional[Dict[str, Any]],
) -> bytes:
    doc = {
        "v": 1,
        "function_fqn": function_fqn,
        "params": params,  # already normalized & sorted
        "inputs": sorted([{"id": h, "role": r} for h, r in inputs], key=lambda x: (x["role"], x["id"])),
        "code": code_hash,
        "env": env_hash,
        "materializers": mats or {},
        "seeds": seeds or {},
    }
    # Canonical JSON: sorted keys, no spaces
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")

def step_signature_hash(payload_bytes: bytes) -> str:
    hasher = Hash()
    return hasher.hash_bytes(payload_bytes)

# --- StepOps + StepContext ---

class StepOps:
    """
    Low-level operations for recording a single step’s lineage and artifacts.

    ``StepOps`` is the imperative API used by step contexts to:
    - register inputs/outputs in the tracking DB,
    - snapshot code/environment,
    - index/copy/link artifacts into the CAS,
    - manage datasets and step status.

    Instances are created by :class:`StepContext` and used internally by
    :class:`StreamStep`. Library users typically do not instantiate this class
    directly; they interact through :meth:`Track.step` / :meth:`Track.stream`.

    Parameters
    ----------
    run_path : Path
        Path to the run directory (``runs/<run_id>``).
    step_hash : str
        Identifier of the step row being mutated.
    cas_policy : {"index", "link", "copy"}, optional
        How to persist artifacts into CAS. Defaults to ``"index"``:
        - ``"index"``: do not copy; record references and metadata.
        - ``"link"``: hard-link into CAS when possible, else copy.
        - ``"copy"``: always copy the file bytes into CAS.

    Attributes
    ----------
    db : Path
        SQLite database path (``nodes/nodes.db``).
    run_path : Path
        Run directory for this step.
    step_hash : str
        Unique identifier of the step this ops object modifies.
    cas_policy : str
        CAS policy in effect for this step (see above).
    """
    def __init__(self, run_path: Path, step_hash: str, cas_policy: str = "index"):
        self.run_path = run_path
        self.step_hash = step_hash
        self.db = run_path / "nodes" / "nodes.db"
        self.cas_policy = cas_policy  # "index" | "link" | "copy"

    def add_input(self, data_hash: str, role: str, arg_name: Optional[str] = None) -> None:
        """
        Record an input edge for this step.

        Parameters
        ----------
        data_hash : str
            Hash of the input artifact (CSV/data/code/env/etc.).
        role : str
            Semantic role of this input (e.g., ``"source"``, ``"seed"``,
            ``"upstream"``, ``"code"``, ``"env"``).
        arg_name : str or None, optional
            Name of the target function argument that receives this input,
            if applicable.

        Returns
        -------
        None
        """
        with _conn(self.db) as c:
            c.execute("""INSERT OR REPLACE INTO step_inputs(step_hash,data_hash,role,arg_name)
                        VALUES(?,?,?,?)""",
                    (self.step_hash, data_hash, role, arg_name))
            c.commit()

    def add_output(self, data_hash: str, name: str) -> None:
        """
        Record an output edge for this step.

        Parameters
        ----------
        data_hash : str
            Hash of the produced artifact.
        name : str
            Logical name of the output (e.g., filename or label).

        Returns
        -------
        None
        """
        with _conn(self.db) as c:
            c.execute("INSERT OR IGNORE INTO step_outputs(step_hash,data_hash,name) VALUES(?,?,?)",
                      (self.step_hash, data_hash, name))
            c.commit()

    def register_data(self, *, kind: str, version: str, path_or_bytes, metadata: Optional[Dict[str, Any]] = None,
                      source_path: Optional[str] = None) -> str:
        """
        Register data in CAS and the tracking DB, returning its content hash.

        If ``path_or_bytes`` is bytes, the payload is written into CAS.
        If it is a path, behavior depends on ``cas_policy``:
        - ``"index"``: do not copy; index the path + metadata.
        - ``"link"``: hard-link into CAS if possible, else copy.
        - ``"copy"``: copy bytes into CAS.

        Also records a materialization row (path on disk) when a source path
        is used, enabling fast future restores.

        Parameters
        ----------
        kind : str
            Logical kind/category (e.g., ``"csv"``, ``"data"``, ``"code"``, ``"env"``).
        version : str
            Logical version for the kind (schema/format hint).
        path_or_bytes : (bytes | bytearray | str | Path)
            Either raw bytes to store or a filesystem path to index/copy/link.
        metadata : dict or None, optional
            Extra metadata to persist alongside the data row.
        source_path : str or None, optional
            Explicit source path to record (normally inferred from ``path_or_bytes``).

        Returns
        -------
        str
            The computed content hash (hex).
        """
        policy = self.cas_policy  # "index" | "link" | "copy"

        cas_root = self.run_path / "cas" / "sha256" / kind

        if isinstance(path_or_bytes, (bytes, bytearray)):
            # always store bytes in CAS
            data_hash, byte_len = CAS.put_bytes(bytes(path_or_bytes), kind=kind)
            cas_path = cas_root / data_hash[:2] / data_hash[2:]
            cas_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cas_path.with_suffix(".tmp")
            tmp.write_bytes(path_or_bytes)
            os.replace(tmp, cas_path)
            src_path_for_db = None
        else:
            p = Path(path_or_bytes)
            data_hash, byte_len = CAS.put_file(p, kind=kind)
            src_path_for_db = str(p)

            if policy == "index":
                # do not copy; only index the path
                pass
            else:
                cas_path = cas_root / data_hash[:2] / data_hash[2:]
                cas_path.parent.mkdir(parents=True, exist_ok=True)
                if policy == "link":
                    try:
                        os.link(str(p), cas_path)
                    except OSError:
                        shutil.copy2(str(p), cas_path)
                elif policy == "copy":
                    shutil.copy2(str(p), cas_path)

            # optional: remember usable materialization
            with _conn(self.db) as c:
                c.execute("""INSERT OR IGNORE INTO materializations(data_hash,path,ts)
                            VALUES(?,?,?)""", (data_hash, src_path_for_db, _now()))
                c.commit()

        with _conn(self.db) as c:
            c.execute("""INSERT OR IGNORE INTO data(data_hash,kind,version,byte_len,source_path,metadata_json)
                        VALUES(?,?,?,?,?,?)""",
                    (data_hash, kind, version, byte_len, src_path_for_db, json.dumps(metadata or {})))
            c.commit()
        return data_hash

    def add_dataset_member(self, dataset_id: str, data_hash: str, ordinal: int) -> None:
        """
        Append an item to a logical dataset (ordered collection of data hashes).

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        data_hash : str
            Member data hash to add.
        ordinal : int
            Position/order of this member within the dataset.

        Returns
        -------
        None
        """
        with _conn(self.db) as c:
            c.execute("INSERT OR IGNORE INTO datasets(dataset_id) VALUES (?)", (dataset_id,))
            c.execute("""INSERT OR IGNORE INTO dataset_members(dataset_id,data_hash,ordinal)
                         VALUES (?,?,?)""", (dataset_id, data_hash, ordinal))
            c.commit()

    def mark_completed(self) -> None:
        """
        Mark this step as completed.

        Notes
        -----
        Idempotent: a previously-completed step remains completed.
        Stamps ``ts_end`` if not already present.
        """
        with _conn(self.db) as c:
            # Don't downgrade an already-completed step, and don't flip explicit failures.
            c.execute("""
                UPDATE steps
                SET status = CASE
                                WHEN status = 'completed' THEN status
                                ELSE 'completed'
                                END,
                    ts_end = COALESCE(ts_end, ?)
                WHERE step_hash = ?
            """, (_now(), self.step_hash))
            c.commit()

    def mark_aborted(self) -> None:
        """
        Mark this step as aborted (unless already completed).

        Notes
        -----
        Does not downgrade a completed step. Stamps ``ts_end``.
        """
        with _conn(self.db) as c:
            c.execute("""
                UPDATE steps
                SET status = CASE
                                WHEN status = 'completed' THEN status
                                ELSE 'aborted'
                                END,
                    ts_end = COALESCE(ts_end, ?)
                WHERE step_hash = ?
            """, (_now(), self.step_hash))
            c.commit()

    def snapshot_callable(self, fn) -> str:
        """
        Snapshot a Python callable into CAS and register it as a ``code`` input.

        Parameters
        ----------
        fn : callable
            Function object to serialize for lineage/provenance.

        Returns
        -------
        str
            Content hash of the code blob, or ``""`` if snapshot fails.
        """
        try:
            blob, meta = build_code_blob(fn)
            code_hash = self.register_data(kind="code", version="1", path_or_bytes=blob, metadata=meta)
            self.add_input(code_hash, role="code")
            return code_hash
        except Exception:
            return ""

    def snapshot_env(self) -> str:
        """
        Snapshot the current environment manifest and register it as an ``env`` input.

        Returns
        -------
        str
            Content hash of the environment manifest, or ``""`` if snapshot fails.
        """
        try:
            blob, meta = build_env_manifest()
            env_hash = self.register_data(kind="env", version="1", path_or_bytes=blob, metadata=meta)
            self.add_input(env_hash, role="env")
            return env_hash
        except Exception:
            return ""

    def add_upstream_from_meta(self, meta: Optional[Dict[str, Any]]) -> None:
        """
        Add upstream inputs from a propagated meta payload.

        Parameters
        ----------
        meta : dict or None
            Metadata containing optional ``"upstream"`` list
            with items like ``{"id": <data_hash>, "role": "seed"}``.

        Returns
        -------
        None
        """
        if not meta: 
            return
        ups = meta.get("upstream")
        if not isinstance(ups, list): 
            return
        for u in ups:
            dh = u.get("id")
            role = u.get("role", "upstream")
            if isinstance(dh, str):
                self.add_input(dh, role=role)

    def remove_inputs_by_role(self, role: str) -> None:
        """
        Delete all inputs for this step matching a given role.

        Parameters
        ----------
        role : str
            Role name to remove (e.g., ``"seed"``).

        Returns
        -------
        None
        """
        with _conn(self.db) as c:
            c.execute("DELETE FROM step_inputs WHERE step_hash=? AND role=?", (self.step_hash, role))
            c.commit()

class StepContext(contextlib.AbstractContextManager[StepOps]):
    """
    Context manager that opens a step, exposes :class:`StepOps`, and seals it on exit.

    Entering the context inserts a row into ``steps`` with status ``"running"``.
    Exiting the context updates status and ``ts_end`` based on outcome:

    - normal exit → ``"completed"``
    - exception (not ``GeneratorExit``) → ``"failed"``
    - ``GeneratorExit`` → ``"aborted"``

    It also computes and stores a canonical signature hash that includes:
    function FQN, normalized parameters, inputs (including code/env if present),
    and optional materials/seeds when available.

    Parameters
    ----------
    run_path : Path
        Path to ``runs/<run_id>`` for DB and CAS roots.
    function_fqn : str
        Fully-qualified function name of the recorded step.
    params : dict
        Parameters to persist as the step’s parameter payload.
    run_id : str or None, optional
        Run identifier; defaults to ``run_path.name`` if omitted.

    Attributes
    ----------
    step_hash : str
        Step identifier (also used as ``attempt_id``).
    ops : StepOps
        Operational surface to record inputs/outputs/metadata.
    """
    def __init__(self, run_path: Path, function_fqn: str, params: Dict[str, Any], run_id: Optional[str] = None):
        self.run_path = run_path
        self.db = run_path / "nodes" / "nodes.db"
        ensure_schema(run_path)
        self.function_fqn = function_fqn
        self.params_json = _normalize_params(params)
        self.ts_start = _now()
        self.step_hash = _attempt_id() 
        self.run_id = run_id or run_path.name
        self.ops = StepOps(run_path, self.step_hash, cas_policy=_cas_policy_default())
        self._code_hash = None
        self._env_hash = None

    def __enter__(self) -> StepOps:
        """
        Insert the step row and return the :class:`StepOps` interface.

        Returns
        -------
        StepOps
            Operational helper bound to this step.
        """
        with _conn(self.db) as c:
            c.execute("""
            INSERT INTO steps(step_hash,run_id,step_id,function_fqn,params_json,seeds_json,ts_start,status,attempt_id)
            VALUES(?,?,?,?,?,?,?,?,?)
            """, (self.step_hash, self.run_id, None, self.function_fqn,
                json.dumps(self.params_json), None, self.ts_start, "running", self.step_hash))
            c.commit()
        return self.ops
    
    def _finalize_signature(self):
        # pull inputs from DB
        with _conn(self.db) as c:
            rows = c.execute("SELECT data_hash, role FROM step_inputs WHERE step_hash=?", (self.step_hash,)).fetchall()
        inputs = [(r[0], r[1]) for r in rows]

        # optional: fetch code/env from inputs by role
        code_hash = next((h for (h, r) in inputs if r == "code"), None)
        env_hash  = next((h for (h, r) in inputs if r == "env"), None)

        payload = canonical_signature_payload(
            function_fqn=self.function_fqn,
            params=self.params_json,
            inputs=inputs,
            code_hash=code_hash,
            env_hash=env_hash,
            mats=None,
            seeds=None,
        )
        sig = step_signature_hash(payload)
        with _conn(self.db) as c:
            # add a column once: ALTER TABLE steps ADD COLUMN sig_hash TEXT;
            c.execute("UPDATE steps SET sig_hash=? WHERE step_hash=?", (sig, self.step_hash))
            c.commit()

    def __exit__(self, exc_type, exc, tb) -> bool:
        """
        Finalize signature and transition step status based on outcome.

        Parameters
        ----------
        exc_type, exc, tb
            Exception triplet provided by the context protocol.

        Returns
        -------
        bool
            Always ``False`` to propagate exceptions to callers.
        """
        self._finalize_signature()
        # Read current status to avoid downgrades
        with _conn(self.db) as c:
            row = c.execute("SELECT status FROM steps WHERE step_hash=?", (self.step_hash,)).fetchone()
            current = row[0] if row else None

        if current == "completed":
            # Already sealed elsewhere (e.g., on last batch) — leave it.
            status = "completed"
        else:
            if exc_type is GeneratorExit:
                status = "aborted"
                exc_type = None
            elif exc_type is None:
                status = "completed"
            else:
                status = "failed"

        with _conn(self.db) as c:
            c.execute("UPDATE steps SET status=?, ts_end=? WHERE step_hash=?",
                    (status, _now(), self.step_hash))
            c.commit()
        return False
    
class StreamStep:
    """
    Manage a single step over a stream of batches.

    Responsibilities
    ----------------
    - Create and hold an underlying :class:`StepContext`.
    - Snapshot code/environment once at construction.
    - For each batch, record upstream lineage and propagate downstream meta.
    - Mark the step completed when the caller flags the last batch.
    - Provide explicit close paths for success or error.

    Parameters
    ----------
    tracker : Track
        The run tracker providing :meth:`Track.step`.
    function_fqn : str
        Fully-qualified function name associated with this stream.
    params : dict
        Parameters persisted for the step.
    code_fn : callable or None, keyword-only
        Optional function to snapshot as the code lineage.

    Attributes
    ----------
    step : StepOps
        Operational surface of the underlying step.
    """
    def __init__(self, tracker, function_fqn: str, params: Dict[str, Any], *, code_fn=None):
        self._cm = tracker.step(function_fqn, params)
        self.step: StepOps = self._cm.__enter__()
        # snapshot code/env once
        if code_fn is not None:
            self.step.snapshot_callable(code_fn)
        self.step.snapshot_env()

    @property
    def step_hash(self) -> str:
        """
        Identifier of the underlying step.

        Returns
        -------
        str
            The step hash.
        """
        return self.step.step_hash

    def emit(self, *, last_batch: bool, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Record per-batch lineage and produce downstream metadata.

        Adds upstream edges from ``meta`` (if any), mirrors caller metadata,
        annotates the current step hash, and, when ``last_batch`` is True,
        seals the step as completed.

        Parameters
        ----------
        last_batch : bool
            Whether this is the final batch for the step.
        meta : dict or None
            Metadata propagated from upstream (may include ``"upstream"`` list).

        Returns
        -------
        dict
            Downstream metadata for the batch (includes ``"producer_step"`` and
            preserves caller-provided ``"ordinal"`` if present).
        """
        # record lineage
        self.step.add_upstream_from_meta(meta)

        # preserve caller meta (hints) and overlay our bookkeeping
        out = dict(meta or {})
        out["producer_step"] = self.step.step_hash
        # keep caller-provided ordinal if present; else pass through whatever you computed
        if "ordinal" not in out and meta and "ordinal" in meta:
            out["ordinal"] = meta["ordinal"]

        if last_batch:
            self.step.mark_completed()
        return out

    def close_ok(self) -> None:
        """
        Close the underlying context as a successful completion.

        Returns
        -------
        None
        """
        self._cm.__exit__(None, None, None)

    def close_error(self, exc_type, exc, tb) -> None:
        """
        Close the underlying context with an error classification.

        Parameters
        ----------
        exc_type, exc, tb
            Exception triplet describing the failure.

        Returns
        -------
        None
        """
        self._cm.__exit__(exc_type, exc, tb)
import uuid
from typing import Dict, Optional, Any, Tuple, List
import contextlib, json, sqlite3, time
from pathlib import Path
import shutil
import os

from blase.utils.hashing import Hash
from blase.tracking.snapshot import build_code_blob, build_env_manifest
from blase.utils.config import cas_policy as _cas_policy_default

# --- CAS and DAG local storage helpers ---

DDL = [
    # runs (optional)
    """CREATE TABLE IF NOT EXISTS runs (
         run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, label TEXT, metadata_json TEXT
       )""",
    # steps
    """CREATE TABLE IF NOT EXISTS steps (
         step_hash TEXT PRIMARY KEY,
         run_id    TEXT NOT NULL,
         step_id   TEXT,
         function_fqn TEXT NOT NULL,
         params_json  TEXT NOT NULL,
         seeds_json   TEXT,
         ts_start     TEXT NOT NULL,
         ts_end       TEXT,
         attempt_id   TEXT,
         sig_hash     TEXT,
         status       TEXT NOT NULL CHECK (status IN ('running','completed','failed','aborted'))
       )""",
    # data blobs
    """CREATE TABLE IF NOT EXISTS data (
         data_hash     TEXT PRIMARY KEY,
         kind          TEXT NOT NULL,
         version       TEXT NOT NULL,
         byte_len      INTEGER,
         source_path   TEXT,
         metadata_json TEXT
       )""",
    # edges
    """CREATE TABLE IF NOT EXISTS step_inputs (
         step_hash TEXT NOT NULL, data_hash TEXT NOT NULL, role TEXT NOT NULL, arg_name TEXT,
         PRIMARY KEY (step_hash, data_hash, role)
       )""",
    """CREATE TABLE IF NOT EXISTS step_outputs (
         step_hash TEXT NOT NULL, data_hash TEXT NOT NULL, name TEXT NOT NULL,
         PRIMARY KEY (step_hash, name),
         UNIQUE (data_hash)
       )""",
    # datasets
    """CREATE TABLE IF NOT EXISTS datasets (
         dataset_id TEXT PRIMARY KEY, name TEXT, metadata_json TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS dataset_members (
         dataset_id TEXT NOT NULL, data_hash TEXT NOT NULL, ordinal INTEGER NOT NULL,
         PRIMARY KEY (dataset_id, ordinal),
         UNIQUE (dataset_id, data_hash)
       )""",
    # materializations (optional)
    """CREATE TABLE IF NOT EXISTS materializations (
         data_hash TEXT NOT NULL, path TEXT NOT NULL, ts TEXT NOT NULL,
         PRIMARY KEY (data_hash, path)
       )""",
    # indexes
    "CREATE INDEX IF NOT EXISTS idx_inputs_data  ON step_inputs(data_hash)",
    "CREATE INDEX IF NOT EXISTS idx_outputs_data ON step_outputs(data_hash)",
    "CREATE INDEX IF NOT EXISTS idx_outputs_step ON step_outputs(step_hash)",
    "CREATE INDEX IF NOT EXISTS idx_steps_sig ON steps(sig_hash)",
    "CREATE INDEX IF NOT EXISTS idx_data_kind    ON data(kind)"
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
    def __init__(self, run_path: Path, step_hash: str, cas_policy: str = "index"):
        self.run_path = run_path
        self.step_hash = step_hash
        self.db = run_path / "nodes" / "nodes.db"
        self.cas_policy = cas_policy  # "index" | "link" | "copy"

    def add_input(self, data_hash: str, role: str, arg_name: Optional[str] = None) -> None:
        with _conn(self.db) as c:
            c.execute("""INSERT OR REPLACE INTO step_inputs(step_hash,data_hash,role,arg_name)
                        VALUES(?,?,?,?)""",
                    (self.step_hash, data_hash, role, arg_name))
            c.commit()

    def add_output(self, data_hash: str, name: str) -> None:
        with _conn(self.db) as c:
            c.execute("INSERT OR IGNORE INTO step_outputs(step_hash,data_hash,name) VALUES(?,?,?)",
                      (self.step_hash, data_hash, name))
            c.commit()

    def register_data(self, *, kind: str, version: str, path_or_bytes, metadata: Optional[Dict[str, Any]] = None,
                      source_path: Optional[str] = None) -> str:
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
        with _conn(self.db) as c:
            c.execute("INSERT OR IGNORE INTO datasets(dataset_id) VALUES (?)", (dataset_id,))
            c.execute("""INSERT OR IGNORE INTO dataset_members(dataset_id,data_hash,ordinal)
                         VALUES (?,?,?)""", (dataset_id, data_hash, ordinal))
            c.commit()

    def mark_completed(self) -> None:
        """Seal this step as completed and stamp ts_end. Idempotent."""
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
        try:
            blob, meta = build_code_blob(fn)
            code_hash = self.register_data(kind="code", version="1", path_or_bytes=blob, metadata=meta)
            self.add_input(code_hash, role="code")
            return code_hash
        except Exception:
            return ""

    def snapshot_env(self) -> str:
        try:
            blob, meta = build_env_manifest()
            env_hash = self.register_data(kind="env", version="1", path_or_bytes=blob, metadata=meta)
            self.add_input(env_hash, role="env")
            return env_hash
        except Exception:
            return ""

    def add_upstream_from_meta(self, meta: Optional[Dict[str, Any]]) -> None:
        if not meta: return
        ups = meta.get("upstream")
        if not isinstance(ups, list): return
        for u in ups:
            dh = u.get("id")
            role = u.get("role", "upstream")
            if isinstance(dh, str):
                self.add_input(dh, role=role)

class StepContext(contextlib.AbstractContextManager[StepOps]):
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
    Manages one StepContext over a stream of batches. Handles:
      - code/env snapshot once
      - per-batch upstream recording (dedup via PK)
      - sealing on last batch
      - proper close on success/error
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
        return self.step.step_hash

    def emit(self, *, last_batch: bool, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Call per batch after you compute outputs.
        Records upstream; seals step if last_batch.
        Returns downstream meta to propagate.
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
        self._cm.__exit__(None, None, None)

    def close_error(self, exc_type, exc, tb) -> None:
        self._cm.__exit__(exc_type, exc, tb)
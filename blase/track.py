from typing import Any, Callable, Dict, Optional, Tuple
from pathlib import Path
import json
import sqlite3
from datetime import datetime
import platform
import sys
import subprocess

from blase.utils.hashing import Hash

class Track:
    """
    Tracks, logs, and manages project-based machine learning runs for reproducibility, modular retracing,
    and metadata storage across the entire `blase` framework.

    `Track` allows users to group disjointed experiments under a shared project and automatically logs
    every main step in the ML workflow (e.g., Extract, Transform, Train, Evaluate, etc.). It supports
    both explicit run control (for power users) and automatic logging behind the scenes.

    Core Features:
    --------------
    - **Project-based structure**: All tracked runs are organized under a common project namespace.
    - **Step-level logging**: Each ML step (prepare, train, etc.) gets its own JSON file for traceability.
    - **Hash-based reproducibility**: Code, arguments, and input references are hashed and stored for recovery.
    - **Custom script capture**: Stores user-defined functions and script files with hash-matched filenames.
    - **Artifact indexing**: Models, preprocessed files, and logs are tied to their originating steps.
    - **Resumable experimentation**: Easily trace previous steps, compare runs, or continue iterations.
    - **Local-first philosophy**: Works entirely offline with optional CLI integration for scripting or agents.

    Method Interface:
    -----------------
    This class is split into two layers:

    - **Public methods** — For users to manage runs, restore past sessions, and compare results.
    - **Protected methods** — Used internally by `blase` modules to automatically log arguments,
      artifacts, scripts, and hashes. Not intended for direct use unless customizing `blase`.

    File Structure:
    ---------------
    When a run is created under a project, it generates:

        /project/
        └── <runs>/
            └── <run_id>_<timestamp>/
                ├── logs/
                │   ├── prepare.blase
                │   ├── train.blase
                ├── scripts/
                │   └── model_def.py
                └── artifacts/
                    └── model_v1.h5

    Each JSON log contains:
    - Step name
    - Timestamp
    - Hashes of inputs, outputs, and scripts
    - References to prior steps (if any)
    - Status flags (e.g., complete, failed)

    Usage:
    ------
    >>> track = Track(project="diabetes_prediction")
    >>> track.start_run()
    >>> track.start_step("prepare", args={"source": "data.csv"})
    >>> track.log_script("prepare_target.py", func=custom_target_function)
    >>> track.log_artifact("prep_data.npy")
    >>> track.finalize_step()

    CLI Integration:
    ----------------
    The `Track` system is compatible with the `blase` CLI for headless workflows:

        $ blase run --project diabetes_prediction --step prepare

    Notes:
    ------
    - Users can manually control the project and run IDs or let them be auto-generated.
    - Each step is tracked independently but can reference prior hashes for full lineage.
    - Works with disjointed scripts and notebook workflows; not limited to full-pipeline execution.

    See Also:
    ---------
    - `Hash` utility for function and config fingerprinting.
    - `Pipeline` for defining and executing full training sequences.
    - `Prepare`, `Train`, `Evaluate`, etc., which use Track's protected methods internally.
    """

    # ──────────────────────────────────────────────────────
    # Public Methods — For Users Managing Runs & Reproducibility
    # ──────────────────────────────────────────────────────

    def __init__(self, project: Optional[str] = None, run_name: Optional[str] = None) -> None:
        """Initialize tracking for a specific project. Creates a new run directory if one doesn't exist."""
        pass

    def start_run(self, run_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Begin a new run under the current project. Sets timestamp, base path, and creates folders."""
        pass

    def end_run(self, status: str = "complete") -> None:
        """Finalize the run, optionally marking it as complete or failed."""
        pass

    def list_runs(self) -> list:
        """List all available runs under the current project."""
        pass

    def load_run(self, run_id: str) -> None:
        """Load metadata and configuration from a previous run. Allows walking back to prior state."""
        pass

    def diff_runs(self, run_id_1: str, run_id_2: str) -> Dict[str, Any]:
        """Compare two runs and return differences in arguments, scripts, and artifacts."""
        pass

    def latest(self, step_name: Optional[str] = None) -> Optional[str]:
        """Return the most recent run ID for the full project or a specific step."""
        pass

    def chain(self, step_order: list) -> list:
        """Return a list of references forming a lineage chain between logged steps."""
        pass

    def restore_step(
            self,
            step_hash
    ):
        """Walk DAG backward and rerun steps if assets are missing."""
        # walk back to last available data node

        # dynamically write restoration functions

        # validate restore

    # ──────────────────────────────────────────────────────
    # Protected Methods — For Use Inside ML Modules (e.g., Prepare, Train)
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_run_directory(track: bool) -> Path:
        """
        Returns the tracking directory as a Path object, or raises an error if not found/created.
        """
        cwd = Path.cwd()
        parent = cwd.parent

        if track:
            if not (cwd / "runs").exists() and not (parent / "runs").exists():
                response = input(f"Tracking directory not found. Create new run directory? [y/N]: ")
                if response.lower() == "y":
                    print("Directories:")
                    print(f"Parent: {parent}")
                    print(f"Current: {cwd}")
                    generate = input("Track in current directory or parent? [C/P]: ")
                    if generate.lower() == "c":
                        (cwd / "runs").mkdir(parents=True, exist_ok=True)
                        return cwd / "runs"
                    else:
                        (parent / "runs").mkdir(parents=True, exist_ok=True)
                        return parent / "runs"
                else:
                    raise RuntimeError("Tracking is enabled but no valid '/runs' directory found.")
            else:
                return cwd / "runs" if (cwd / "runs").exists() else parent / "runs"
        else:
            # If tracking is off but dir exists, fallback
            if (cwd / "runs").exists():
                return cwd / "runs"
            elif (parent / "runs").exists():
                return parent / "runs"
            else:
                raise RuntimeError("Tracking disabled and no existing '/runs' directory found.")
            
    @staticmethod
    def __initialize_run_structure(track_path: Path, run_id: str) -> Path:
        """Create /steps, /assets, and initialize run files."""
        run_path = track_path / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        (run_path / "assets").mkdir(exist_ok=True)
        (run_path / "nodes").mkdir(exist_ok=True)

        # Create nodes and dag sqlite databases
        node_db_path = run_path / "nodes" / "nodes.db"
        create_steps_table = """
        CREATE TABLE IF NOT EXISTS steps (
            step_hash TEXT PRIMARY KEY,
            step_id TEXT UNIQUE,
            function TEXT,
            parent TEXT,
            parent_type TEXT,
            params JSON,
            timestamp_start TEXT,
            timestamp_end TEXT,
            status TEXT,
            outputs JSON
        );
        """
        create_data_table = """
        CREATE TABLE IF NOT EXISTS data (
            data_hash TEXT PRIMARY KEY,
            source_path TEXT,
            parent TEXT,
            parent_type TEXT,
            logged_by TEXT,
            metadata JSON
        );
        """
        create_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_step_parent ON steps(parent);",
            "CREATE INDEX IF NOT EXISTS idx_data_logged_by ON data(logged_by);",
            "CREATE INDEX IF NOT EXISTS idx_data_parent ON data(parent);"
        ]
        with sqlite3.connect(node_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(create_steps_table)
            cursor.execute(create_data_table)
            for idx_sql in create_indexes:
                cursor.execute(idx_sql)
            conn.commit()

        # Global metadata
        (run_path / "run_metadata.blase").write_text(json.dumps({
            "created_at": datetime.now().isoformat(),
            "system": platform.system(),
            "platform": platform.platform(),
            "python_version": sys.version,
        }, indent=2))

        # Optional: Freeze environment (fails silently if pip not present)
        try:
            with open(run_path / "dependencies.txt", "w") as f:
                subprocess.run(["pip", "freeze"], stdout=f, check=True)
        except Exception as e:
            print(f"[blase] Warning: Could not freeze dependencies: {e}")

        return run_path

    @staticmethod
    def _validate_active_run(track_dir: Path) -> Path:
        """Check run directory for active run, or create a new one."""
        track_path = Path(track_dir)
        active_path = track_path / "active_run.blase"

        if not active_path.exists():
            response = input("Start tracking your first run? [y/N]: ")
            if response.lower() == "y":
                run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                run_path = Track.__initialize_run_structure(track_path, run_id)

                # Write active run pointer
                with open(active_path, "w") as f:
                    json.dump({"run_id": run_id, "status": "active"}, f, indent=2)

                return run_path
            else:
                raise RuntimeError("No active run found. Please start a new run.")

        # Resume existing run
        with open(active_path, "r") as f:
            data = json.load(f)

        run_id = data.get("run_id")
        if not run_id:
            raise RuntimeError("Corrupted active_run.blase — missing run_id.")

        run_path = track_path / run_id
        if not run_path.exists():
            raise RuntimeError(f"Run folder {run_path} not found.")

        return run_path

    @staticmethod
    def _start_step(
        run_path: Path,
        function: str,
        params: Dict[str, Any],
        parent: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Starts a new tracked step within a run directory.

        Args:
            run_path: Path to the active run directory.
            function: Name of the function being tracked.
            params: Dictionary of parameters passed to the function.
            parent: Step or data hash.

        Returns:
            step_id: The unique identifier for the step.
            step_hash: Hash of function, params, and step number.
        """
        node_db_path = run_path / "nodes" / "nodes.db"
        # Define values
        with sqlite3.connect(node_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM steps;")
            count = cursor.fetchone()[0]

        hasher = Hash()
        step_hash = hasher.hash_object({
            "function": function,
            "params": params,
            "step_num": count
        })
        step_id = f"step_{count}_{function.split('.')[-1]}"
        timestamp_start = datetime.now().isoformat()
        timestamp_end = None
        status = "pending"
        outputs = {}

        # Add entry
        new_entry = """
        INSERT INTO steps (
            step_hash, step_id, function,
            parent, params, timestamp_start,
            timestamp_end, status, outputs
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = (
            step_hash, step_id, function,
            parent, json.dumps(params), timestamp_start,
            timestamp_end, status, json.dumps(outputs) 
        )
        with sqlite3.connect(node_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(new_entry, values)
            conn.commit()

        return step_id, step_hash
    
    @staticmethod
    def _end_step(
        step_hash: Optional[str],
        run_path: Optional[Path],
        status: str = "completed",
        outputs: Optional[Dict[str, Any]] = None,
        parent: Optional[tuple[str, str]] = None,
    ) -> None:
        """
        Finalizes a tracked step by updating its status and outputs.

        Args:
            step_hash: String of step hash returned by _start_step
            status: Step status ("completed", "failed", etc.).
            outputs: Optional dictionary of output metadata or hashes.
            file_hash: Optionally update a file's hash in log
        """
        if run_path is not None:
            node_db_path = run_path / "nodes" / "nodes.db"
            # Define values
            end_time = datetime.now().isoformat()
            parent_hash, parent_type = parent

            # Update table
            query = """
            UPDATE steps
            SET status = ?, timestamp_end = ?, outputs = ?, parent = ?, parent_type = ?
            WHERE step_hash = ?
            """
            values = (
                status, end_time, json.dumps(outputs), parent_hash, parent_type, step_hash
            )
            with sqlite3.connect(node_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()

    @staticmethod
    def _log_data(
        run_path: Path,
        file_hash: str,
        parent_dict: Dict,
        metadata: Optional[dict] = None
    ):
        """
        Logs metadata for a data file identified by its hash into a structured JSON lookup file.

        If the file hash does not already exist in the lookup file, a new entry is created with
        the provided metadata. The method expects a specific directory structure and file format.

        Args:
            run_path (Path): Path to the root directory of the current run.
            file_hash (str): Unique hash representing the data file to be logged.
            parent_dict (Dict): Metadata dictionary to associate with the file hash.
                                           If None or empty, defaults to an empty dictionary.

        Raises:
            FileNotFoundError: If `run_path` or the data lookup file does not exist.
        """
        node_db_path = run_path / "nodes" / "nodes.db"

        if not node_db_path.exists():
            raise FileNotFoundError(f"The node database does not exist: {node_db_path}")

        # Define values
        source_path = str(parent_dict['source_path'])
        parent = parent_dict['parent']
        parent_type = parent_dict['parent_type']
        logged_by = parent_dict['logged_by']
        metadata_json = json.dumps(metadata or {})

        # Add entry
        with sqlite3.connect(node_db_path) as conn:
            cursor = conn.cursor()

            # Check if this hash exists
            cursor.execute("SELECT 1 FROM data WHERE data_hash = ?", (file_hash,))
            exists = cursor.fetchone()

            if not exists:
                query = """
                INSERT INTO data(
                    data_hash,
                    source_path,
                    parent,
                    parent_type,
                    logged_by,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """
                values = (
                    file_hash,
                    source_path,
                    parent,
                    parent_type,
                    logged_by,
                    metadata_json
                )
                cursor.execute(query, values)
                conn.commit()

    def _log_script(self, script_path: str, func: Optional[Callable] = None) -> None:
        """Hash and save a user-defined script or function source code into the run."""
        pass

    def _log_args(self, args: Dict[str, Any]) -> None:
        """Save all step-level parameters (with hash for reproducibility)."""
        pass

    def _log_artifact(self, artifact_path: str) -> None:
        """Register saved data or model artifacts inside the run."""
        pass

    def _finalize_step(self, status: str = "complete") -> None:
        """Close out the current step and write the log to disk."""
        pass

    def _get_step_log_path(self, step_name: str) -> str:
        """Return the filepath where the current step log should be written."""
        pass

    def _ensure_directories(self) -> None:
        """Create /logs, /scripts, /artifacts folders if they do not exist."""
        pass

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Hash a config dictionary for reproducibility and file naming."""
        pass

    def _hash_function(self, func: Callable) -> str:
        """Hash the source code of a user-defined function."""
        pass

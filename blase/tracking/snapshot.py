from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import inspect
import json
import textwrap
from pathlib import Path


def _read_module_source(mod) -> Optional[str]:
    try:
        p = getattr(mod, "__file__", None)
        if p and (p.endswith(".py") or p.endswith(".pyw")) and Path(p).exists():
            return Path(p).read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        return inspect.getsource(mod)
    except Exception:
        return None


def _read_function_source(fn) -> Optional[str]:
    try:
        return textwrap.dedent(inspect.getsource(fn))
    except Exception:
        return None


def build_code_blob(fn: Any) -> Tuple[bytes, Dict[str, Any]]:
    """
    Build a serialized code snapshot for a given function.

    This function captures the structural and source-level
    information of a Python function along with its defining module.
    It produces a canonical JSON-encoded blob suitable for
    content-addressable storage (CAS) and a companion metadata
    dictionary for quick inspection.

    Parameters
    ----------
    fn : Any
        Python function (or callable object) to snapshot. The function
        must be importable and have an associated module.

    Returns
    -------
    Tuple[bytes, Dict[str, Any]]
        - **blob** : bytes
            Canonical JSON-encoded representation of the snapshot.
            Includes:

            * `v`: Schema version.
            * `entry`: Dict with `module` name and function
              `qualname`.
            * `module_source`: Source text of the defining module,
              if available.
            * `function_source`: Extracted source code of the function,
              if available.

        - **meta** : dict
            Companion metadata for lightweight inspection.
            Includes:

            * `v`: Schema version.
            * `entry`: Same entry dict (module and qualname).
            * `has_module`: Boolean indicating if module source
              was captured.
            * `has_function`: Boolean indicating if function source
              was captured.

    Notes
    -----
    - The blob is normalized with sorted keys and compact separators,
      ensuring reproducible hashing.
    - Capturing both module and function source enables replay or
      re-materialization of steps even if the live code changes.
    - If the module source cannot be read (e.g., builtins or C
      extensions), `module_source` will be `None`.

    Examples
    --------
    >>> def foo(x): return x + 1
    >>> blob, meta = build_code_blob(foo)
    >>> isinstance(blob, bytes)
    True
    >>> meta["entry"]["qualname"]
    'foo'
    >>> meta["has_function"]
    True
    """
    mod = inspect.getmodule(fn)
    doc = {
        "v": 1,
        "entry": {
            "module": getattr(mod, "__name__", None),
            "qualname": getattr(fn, "__qualname__", getattr(fn, "__name__", None)),
        },
        "module_source": _read_module_source(mod),
        "function_source": _read_function_source(fn),
    }
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    meta = {
        "v": 1,
        "entry": doc["entry"],
        "has_module": doc["module_source"] is not None,
        "has_function": doc["function_source"] is not None,
    }
    return blob, meta


def build_env_manifest() -> Tuple[bytes, Dict[str, Any]]:
    """
    Build a serialized manifest of the current Python environment.

    This function inspects installed distributions (via
    ``importlib.metadata.distributions``) and produces a
    canonical JSON-encoded manifest of package names and versions.
    The result is suitable for environment reproducibility,
    dependency tracking, and content-addressable storage.

    Parameters
    ----------
    None

    Returns
    -------
    Tuple[bytes, Dict[str, Any]]
        - **blob** : bytes
            Canonical JSON-encoded representation of the environment
            manifest. Includes:

            * `v`: Schema version.
            * `packages`: List of dicts with fields
              `{"name": str, "version": str}`.

        - **meta** : dict
            Companion metadata with summary information. Includes:

            * `v`: Schema version.
            * `count`: Number of packages captured.

    Notes
    -----
    - Package list is sorted case-insensitively by name and then by
      version for deterministic output.
    - Compatible with Python 3.9+ using the standard library; falls
      back to `importlib_metadata` on older versions.
    - If a package does not expose `Name` or `Version` metadata, it
      is skipped.

    Examples
    --------
    >>> blob, meta = build_env_manifest()
    >>> isinstance(blob, bytes)
    True
    >>> meta["count"] > 0
    True
    >>> "numpy" in [p["name"].lower() for p in json.loads(blob.decode())["packages"]]
    True
    """
    try:
        from importlib.metadata import distributions  # py39 ok via backport if needed
    except Exception:
        from importlib_metadata import distributions  # type: ignore
    pkgs: List[Dict[str, str]] = []
    for d in distributions():
        name = getattr(d, "metadata", {}).get("Name") or d.metadata.get("Name")  # type: ignore[attr-defined]
        ver = getattr(d, "version", None) or d.metadata.get("Version")  # type: ignore[attr-defined]
        if name and ver:
            pkgs.append({"name": name, "version": ver})
    pkgs.sort(key=lambda x: (x["name"].lower(), x["version"]))
    doc = {"v": 1, "packages": pkgs}
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    meta = {"v": 1, "count": len(pkgs)}
    return blob, meta

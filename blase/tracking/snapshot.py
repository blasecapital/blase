from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import inspect, json, textwrap, sys
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
    meta = {"v": 1, "entry": doc["entry"], "has_module": doc["module_source"] is not None,
            "has_function": doc["function_source"] is not None}
    return blob, meta

def build_env_manifest() -> Tuple[bytes, Dict[str, Any]]:
    try:
        from importlib.metadata import distributions  # py39 ok via backport if needed
    except Exception:
        from importlib_metadata import distributions  # type: ignore
    pkgs: List[Dict[str,str]] = []
    for d in distributions():
        name = getattr(d, "metadata", {}).get("Name") or d.metadata.get("Name")  # type: ignore[attr-defined]
        ver  = getattr(d, "version", None) or d.metadata.get("Version")          # type: ignore[attr-defined]
        if name and ver:
            pkgs.append({"name": name, "version": ver})
    pkgs.sort(key=lambda x: (x["name"].lower(), x["version"]))
    doc = {"v": 1, "packages": pkgs}
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    meta = {"v": 1, "count": len(pkgs)}
    return blob, meta
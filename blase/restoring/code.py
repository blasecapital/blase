from __future__ import annotations
from pathlib import Path
import json, types

def load_callable_from_blob(blob_path: Path):
    """
    Load a callable from a snapshot blob (built by snapshot.build_code_blob).
    Safe: executes code under a synthetic module name (not __main__).
    Robust: handles <locals> in qualname and resolves from module/globals.
    """
    data = json.loads(Path(blob_path).read_text("utf-8"))
    entry = data.get("entry") or {}
    qualname = (entry.get("qualname") or "").strip()
    mod_hint = (entry.get("module") or "blase_restored.anon").strip()

    module_source = data.get("module_source") or ""
    function_source = data.get("function_source") or ""

    mod_name = f"blase_restored.{mod_hint}"
    mod = types.ModuleType(mod_name)
    g = mod.__dict__
    g["__name__"] = mod_name
    g["__package__"] = mod_name.rpartition(".")[0]

    # Define module-level symbols first (helpers/imports)
    if module_source:
        exec(module_source, g)

    # Ensure the function exists even if it wasn't top-level in the module source
    if function_source:
        exec(function_source, g)

    # Try to resolve the callable:
    #  - prefer walking attributes on the module object
    #  - ignore "<locals>" segments
    #  - fall back to the final name in globals
    parts = [p for p in qualname.split(".") if p and p != "<locals>"]
    if parts:
        # first try attribute walk on the module
        obj = mod
        try:
            for part in parts:
                obj = getattr(obj, part)
            return obj
        except Exception:
            # fall back to final name in globals
            name = parts[-1]
            if name in g:
                return g[name]

    # As a last resort, look for any callable in globals with the same name as the function_source defines
    # (best effort; avoids total failure if qualname is odd)
    # Try to extract a name from the first line of function_source (e.g., "def foo(")
    try:
        line0 = function_source.strip().splitlines()[0]
        if line0.startswith("def ") and "(" in line0:
            cand = line0[4: line0.index("(")].strip()
            if cand in g and callable(g[cand]):
                return g[cand]
    except Exception:
        pass

    raise AttributeError(f"Could not resolve callable from blob (qualname='{qualname}', module='{mod_hint}').")
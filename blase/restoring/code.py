from __future__ import annotations
from pathlib import Path
import json
import types
import importlib.machinery
import sys


def load_callable_from_blob(blob_path: Path):
    """
    Load a Python callable from a serialized code snapshot blob.

    This function reconstructs a callable that was previously stored with
    ``snapshot.build_code_blob``. It executes the stored module and function
    source under a synthetic module namespace (not ``__main__``), and then
    attempts to resolve the recorded callable by walking its qualified name.
    It also gracefully handles cases where the function was originally defined
    inside a closure (``<locals>``) or where qualified name resolution fails,
    by falling back to globals or parsing the function source.

    Parameters
    ----------
    blob_path : Path
        Path to the JSON blob file produced by ``snapshot.build_code_blob``.
        The file is expected to contain:

        - ``entry``: metadata about the callable (qualname, module).
        - ``module_source``: source code for helper symbols and imports.
        - ``function_source``: source code for the target callable.

    Returns
    -------
    callable
        The reconstructed Python function, method, or callable object
        restored from the blob.

    Raises
    ------
    AttributeError
        If the callable cannot be resolved from the snapshot.
    json.JSONDecodeError
        If the blob file is not valid JSON.
    OSError
        If the blob file cannot be read.

    Notes
    -----
    Resolution strategy:
    1. Create a synthetic module named ``blase_restored.<mod_hint>``.
    2. Execute ``module_source`` in its globals to establish context.
    3. Execute ``function_source`` to define the target callable.
    4. Attempt to resolve the callable by walking ``qualname`` as attributes.
       - ``<locals>`` segments are ignored to tolerate nested function names.
    5. If that fails, fall back to looking up the last name segment in globals.
    6. As a last resort, parse the first line of ``function_source`` to extract
       the defined function name and check for a callable in globals.

    Security
    --------
    The blob is executed with ``exec`` under a synthetic namespace. While it
    avoids polluting ``__main__``, loading untrusted blobs may still execute
    arbitrary code and should be avoided in hostile contexts.

    Examples
    --------
    >>> path = Path("snapshots/my_func.blob.json")
    >>> fn = load_callable_from_blob(path)
    >>> fn(5)
    42
    """
    data = json.loads(Path(blob_path).read_text("utf-8"))
    entry = data.get("entry") or {}
    qualname = (entry.get("qualname") or "").strip()
    mod_hint = (entry.get("module") or "blase_restored.anon").strip()

    module_source = data.get("module_source") or ""
    function_source = data.get("function_source") or ""

    # Create a real module object
    mod_name = f"blase_restored.{mod_hint}"
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(blob_path)
    mod.__package__ = mod_name.rpartition(".")[0] or None
    mod.__spec__ = importlib.machinery.ModuleSpec(
        name=mod_name, loader=None, origin=str(blob_path)
    )

    # Register module before exec so dataclasses/typing can resolve it
    sys.modules[mod_name] = mod
    # Also register parent package if absent
    pkg = mod.__package__
    if pkg and pkg not in sys.modules:
        pkg_mod = types.ModuleType(pkg)
        pkg_mod.__path__ = []  # namespace pkg
        sys.modules[pkg] = pkg_mod

    g = mod.__dict__

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
            cand = line0[4 : line0.index("(")].strip()
            if cand in g and callable(g[cand]):
                return g[cand]
    except Exception:
        pass

    raise AttributeError(
        f"Could not resolve callable from blob (qualname='{qualname}', module='{mod_hint}')."
    )

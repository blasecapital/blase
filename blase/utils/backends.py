_HAS_POLARS = False
_HAS_PANDAS = False
_HAS_PYARROW = False

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    pass

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pass

try:
    import pyarrow as pa
    _HAS_PYARROW = True
except ImportError:
    pass


def is_available(lib: str) -> bool:
    """
    Check if a backend library is available.

    Args:
        lib (str): One of 'pandas', 'polars', or 'pyarrow'.

    Returns:
        bool: True if available, False otherwise.
    """
    if lib == "polars":
        return _HAS_POLARS
    elif lib == "pandas":
        return _HAS_PANDAS
    elif lib == "pyarrow":
        return _HAS_PYARROW
    else:
        raise ValueError(f"Unknown backend library: {lib}")


def resolve_backend(preferred: str = "auto") -> str:
    """
    Resolve which backend should be used.

    Args:
        preferred (str): User-specified preference. If 'auto', defaults to polars if available.

    Returns:
        str: The name of the backend ('polars' or 'pandas').
    """
    if preferred == "auto":
        if _HAS_POLARS:
            return "polars"
        elif _HAS_PANDAS:
            return "pandas"
        else:
            raise ImportError("Neither 'polars' nor 'pandas' is installed.")
    elif preferred in ("polars", "pandas"):
        if is_available(preferred):
            return preferred
        raise ImportError(f"Backend '{preferred}' is not available. Please install it.")
    else:
        raise ValueError(f"Invalid backend choice: '{preferred}'")
    
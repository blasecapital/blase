import importlib.util

from typing import Optional, Literal

def _has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

_HAS_POLARS  = _has_package("polars")
_HAS_PANDAS  = _has_package("pandas")
_HAS_PYARROW = _has_package("pyarrow")
_HAS_PILLOW  = _has_package("PIL")
_HAS_CV2 = _has_package("cv2")

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


def resolve_backend_csv(preferred: str = "auto") -> str:
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
    

def resolve_backend_images(backend: Optional[str]) -> Literal["pil", "cv2"]:
    """
    Validate and normalize the image decoding backend.

    Parameters
    ----------
    backend : {"pil", "pillow", "cv2", "opencv", "opencv-python"} or None
        Requested backend name (case-insensitive). If None/empty, defaults to "pil".

    Returns
    -------
    {"pil", "cv2"}
        Canonical backend key used by downstream dispatch.

    Raises
    ------
    ValueError
        If the backend name is not recognized.
    RuntimeError
        If the requested backend is not installed, with an actionable install hint.
    """
    name = (backend or "pil").strip().lower()
    aliases = {
        "pil": "pil",
        "pillow": "pil",
        "cv2": "cv2",
        "opencv": "cv2",
        "opencv-python": "cv2",
    }
    if name not in aliases:
        raise ValueError(f"Invalid backend choice: {backend!r}. "
                         "Use one of: 'pil'/'pillow', 'cv2'/'opencv'.")

    resolved = aliases[name]

    if resolved == "pil":
        if not _HAS_PILLOW:
            raise RuntimeError(
                "Pillow is not installed. Install with: pip install pillow"
            )
    else:  # "cv2"
        if not _HAS_CV2:
            raise RuntimeError(
                "OpenCV (cv2) is not installed. Install with: pip install opencv-python"
            )

    return resolved
    
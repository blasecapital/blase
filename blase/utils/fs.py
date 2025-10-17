from pathlib import Path


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: Path) -> Path:
    """
    Ensure the parent directory of `path` exists. Return the parent Path.
    """
    if not isinstance(path, Path):
        path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    return parent

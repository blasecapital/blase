from pathlib import Path
from typing import Literal

def resolve_conflict_path(path: Path, policy: Literal["rename","overwrite","fail"]="rename",
                          suffix: str="_restored") -> Path:
    if not path.exists():
        return path
    if policy == "overwrite":
        return path
    if policy == "fail":
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    # rename
    base = path.with_suffix("")
    ext = path.suffix
    candidate = base.with_name(base.name + suffix).with_suffix(ext)
    i = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}{suffix}-{i}").with_suffix(ext)
        i += 1
    return candidate
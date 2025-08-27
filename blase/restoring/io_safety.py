from pathlib import Path
from typing import Literal

def resolve_conflict_path(path: Path, policy: Literal["rename","overwrite","fail"]="rename",
                          suffix: str="_restored") -> Path:
    """
    Resolve conflicts when materializing a file to disk.

    This function enforces a conflict-resolution policy when a target
    file path already exists. It is typically used when restoring artifacts
    or outputs into a directory that may already contain prior files.

    Depending on the specified ``policy``, the function will:
    - Return the original path unchanged (if no conflict).
    - Allow overwrite.
    - Fail immediately.
    - Or generate a new unique path by appending a suffix.

    Parameters
    ----------
    path : Path
        The candidate file path where a new artifact should be written.
    policy : {"rename", "overwrite", "fail"}, default="rename"
        Strategy to use if ``path`` already exists:

        - ``"rename"`` : Append ``suffix`` to the filename. If a conflict
          still exists, append an incrementing counter until a free path
          is found. (e.g., ``file_restored.txt``, ``file_restored-1.txt``).
        - ``"overwrite"`` : Return the original path, allowing the caller
          to overwrite the existing file.
        - ``"fail"`` : Raise ``FileExistsError`` to prevent overwrite.
    suffix : str, default="_restored"
        String to append to the base filename when ``policy="rename"``.
        The suffix is inserted before the file extension.

    Returns
    -------
    Path
        A resolved, non-conflicting file path based on the chosen policy.

    Raises
    ------
    FileExistsError
        If ``policy="fail"`` and the file already exists.

    Notes
    -----
    - The rename policy ensures deterministic placement by retrying with
      incremented counters until a free path is available.
    - This function only returns a path; it does not create, delete, or
      modify files on disk.
    - For reproducibility, the same input path and policy will always
      yield the same output if no conflict exists.

    Examples
    --------
    >>> from pathlib import Path
    >>> path = Path("output.csv")

    Case 1: Path does not exist
    >>> resolve_conflict_path(path)
    PosixPath('output.csv')

    Case 2: Path exists, rename policy
    >>> resolve_conflict_path(path, policy="rename")
    PosixPath('output_restored.csv')

    Case 3: Path exists, overwrite policy
    >>> resolve_conflict_path(path, policy="overwrite")
    PosixPath('output.csv')

    Case 4: Path exists, fail policy
    >>> resolve_conflict_path(path, policy="fail")
    Traceback (most recent call last):
        ...
    FileExistsError: Refusing to overwrite existing file: output.csv
    """
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
import os
from pathlib import Path

def cas_policy() -> str:
    # "index" (default) | "link" | "copy"
    return os.getenv("BLASE_CAS_POLICY", "index")

# Discover project root (same folder that contains /runs or /data)
def _project_root() -> Path:
    cwd = Path.cwd()
    for p in (cwd, cwd.parent):
        if (p / "runs").exists() or (p / "data").exists():
            return p
    return cwd  # fallback

PROJECT_ROOT = _project_root()

# Where users keep durable inputs
DATA_SOURCE_OF_TRUTH = PROJECT_ROOT / "data" / "source_of_truth"

# Where users *work* with outputs/derived files
DATA_WORKING = PROJECT_ROOT / "data" / "working"

# Where restored artifacts land by default
RESTORE_DEFAULT_DIR = PROJECT_ROOT / "data" / "restored"

# CAS policy for *data* when registering: "index" (no copy), "link", or "copy"
CAS_POLICY = "index"

# Restore write conflict behavior: "rename" | "overwrite" | "fail"
RESTORE_CONFLICT = "rename"
RESTORE_SUFFIX = "_restored"
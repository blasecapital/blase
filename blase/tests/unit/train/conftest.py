import sys
import types
import importlib.util
from pathlib import Path

import os

# Hide GPUs and cut noisy logs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # or "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# /workspace/blase/tests/unit/train/conftest.py
# parents[3] == /workspace/blase  (your package root)
PKG_ROOT = Path(__file__).resolve().parents[3]
PKG = PKG_ROOT  # <-- was PKG_ROOT / "blase", remove the extra segment


def _seed_pkg():
    if "blase" not in sys.modules:
        m = types.ModuleType("blase")
        m.__path__ = [str(PKG)]
        sys.modules["blase"] = m
    if "blase.training" not in sys.modules:
        m = types.ModuleType("blase.training")
        m.__path__ = [str(PKG / "training")]
        sys.modules["blase.training"] = m
    if "blase.training.defaults" not in sys.modules:
        m = types.ModuleType("blase.training.defaults")
        m.__path__ = [str(PKG / "training" / "defaults")]
        sys.modules["blase.training.defaults"] = m


def import_qualified(modname: str, filepath: Path):
    assert filepath.exists(), f"Missing module file: {filepath}"
    spec = importlib.util.spec_from_file_location(modname, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_seed_pkg()
import_qualified(
    "blase.training.defaults.registry", PKG / "training" / "defaults" / "registry.py"
)
import_qualified(
    "blase.training.defaults.tf_adapters",
    PKG / "training" / "defaults" / "tf_adapters.py",
)
import_qualified(
    "blase.training.defaults.tf_input", PKG / "training" / "defaults" / "tf_input.py"
)

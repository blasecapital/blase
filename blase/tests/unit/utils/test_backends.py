import pytest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from blase.utils import backends

def test_is_available_known_libs():
    # Should not raise errors
    assert isinstance(backends.is_available("pandas"), bool)
    assert isinstance(backends.is_available("polars"), bool)
    assert isinstance(backends.is_available("pyarrow"), bool)

def test_is_available_unknown_lib():
    with pytest.raises(ValueError):
        backends.is_available("unknown_lib")

def test_resolve_backend_auto():
    resolved = backends.resolve_backend("auto")
    assert resolved in ("pandas", "polars")

def test_resolve_backend_manual():
    if backends.is_available("pandas"):
        assert backends.resolve_backend("pandas") == "pandas"
    if backends.is_available("polars"):
        assert backends.resolve_backend("polars") == "polars"

def test_resolve_backend_invalid():
    with pytest.raises(ValueError):
        backends.resolve_backend("not_a_backend")

def test_resolve_backend_unavailable():
    # Simulate a backend that is definitely not installed
    with pytest.raises(ValueError):
        backends.resolve_backend("notInstalled")
        
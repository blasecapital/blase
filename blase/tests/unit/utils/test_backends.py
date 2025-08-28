import pytest
import warnings

from blase.utils import backends

warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_is_available_known_libs():
    # Should not raise errors
    assert isinstance(backends.is_available("pandas"), bool)
    assert isinstance(backends.is_available("polars"), bool)
    assert isinstance(backends.is_available("pyarrow"), bool)

def test_is_available_unknown_lib():
    with pytest.raises(ValueError):
        backends.is_available("unknown_lib")

def test_resolve_backend_csv_auto():
    resolved = backends.resolve_backend_csv("auto")
    assert resolved in ("pandas", "polars")

def test_resolve_backend_csv_manual():
    if backends.is_available("pandas"):
        assert backends.resolve_backend_csv("pandas") == "pandas"
    if backends.is_available("polars"):
        assert backends.resolve_backend_csv("polars") == "polars"

def test_resolve_backend_csv_invalid():
    with pytest.raises(ValueError):
        backends.resolve_backend_csv("not_a_backend")

def test_resolve_backend_csv_unavailable():
    # Simulate a backend that is definitely not installed
    with pytest.raises(ValueError):
        backends.resolve_backend_csv("notInstalled")
        
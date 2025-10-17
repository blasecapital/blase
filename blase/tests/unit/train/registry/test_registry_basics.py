import pytest
from blase.training.defaults import registry


def test_registry_unknown_backend_errors():
    with pytest.raises(NotImplementedError):
        registry.strategy("unknown")  # type: ignore


def test_registry_tf_factories_types():
    strat = registry.strategy("tensorflow")
    prec = registry.precision("tensorflow")
    comp = registry.compiler("tensorflow")
    fitr = registry.fitter("tensorflow")
    cbs = registry.callbacks("tensorflow")
    assert hasattr(strat, "scope")
    assert hasattr(prec, "apply")
    assert hasattr(comp, "compile")
    assert hasattr(fitr, "fit")
    assert hasattr(cbs, "build")

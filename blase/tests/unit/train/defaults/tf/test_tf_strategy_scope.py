import pytest
from blase.training.defaults.registry import strategy

tf = pytest.importorskip("tensorflow")


def test_strategy_scope_context_manager():
    strat = strategy("tensorflow")
    with strat.scope():
        v = tf.Variable(1.0)
    assert float(v.numpy()) == 1.0

import pytest
from blase.train import Train


def test_from_tfrecords_validates_inputs():
    t = Train()
    with pytest.raises(ValueError):
        t.from_tfrecords(train_glob="", feature_spec={})

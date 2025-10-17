import pytest
from blase.training.defaults.registry import tf_dataset_builder


def test_builder_raises_on_empty_glob(tmp_path):
    with pytest.raises(FileNotFoundError):
        tf_dataset_builder(
            train_glob=str(tmp_path / "nope/*.tfrecord"),
            feature_spec={"x": object()},
            parse_fn=None,
            augment_fn=None,
            batch_size=2,
            shuffle_buffer=8,
            repeat_train=False,
            compression=None,
            cache=False,
            snapshot_dir=None,
            drop_remainder=True,
        ).build()

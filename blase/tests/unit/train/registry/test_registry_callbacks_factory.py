import pytest
from blase.training.defaults.registry import callbacks

tf = pytest.importorskip("tensorflow")


def test_callbacks_builds_minimal_set(tmp_path):
    cbs = callbacks(
        "tensorflow",
        checkpoint={"dir": str(tmp_path), "save_best_only": True},
        tensorboard={"log_dir": str(tmp_path / "tb")},
        early_stopping={"monitor": "val_loss", "patience": 1},
    ).build()
    # At least 2 items when tb+ckpt+es present
    assert len(cbs) >= 2

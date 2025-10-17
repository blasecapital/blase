from typing import Any, Dict, Iterable, Optional, Union
import tensorflow as tf
from blase.training.protocols import DatasetBuilder

AUTOTUNE = tf.data.AUTOTUNE


class TfDatasetBuilder(DatasetBuilder):
    def __init__(
        self,
        *,
        train_glob: Union[str, Iterable[str]],
        val_glob: Optional[Union[str, Iterable[str]]] = None,
        test_glob: Optional[Union[str, Iterable[str]]] = None,
        feature_spec: Dict[str, Any],
        parse_fn=None,
        augment_fn=None,
        batch_size: int = 256,
        shuffle_buffer: int = 10_000,
        repeat_train: bool = True,
        compression: Optional[str] = None,
        cache: Union[bool, str] = False,
        snapshot_dir: Optional[str] = None,
        drop_remainder: bool = True,
    ):
        self.cfg = locals().copy()
        self.cfg.pop("self")

    def _read(self, files, training: bool):
        if not files:
            return None
        comp = self.cfg["compression"]
        ds = tf.data.TFRecordDataset(
            files, compression_type=comp or "", num_parallel_reads=AUTOTUNE
        )
        ds = ds.map(
            lambda ex: tf.io.parse_single_example(ex, self.cfg["feature_spec"]),
            num_parallel_calls=AUTOTUNE,
        )
        if self.cfg["parse_fn"]:
            ds = ds.map(self.cfg["parse_fn"], num_parallel_calls=AUTOTUNE)
        if training and self.cfg["augment_fn"]:
            ds = ds.map(self.cfg["augment_fn"], num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.shuffle(self.cfg["shuffle_buffer"], reshuffle_each_iteration=True)
            if self.cfg["repeat_train"]:
                ds = ds.repeat()
        if self.cfg["cache"]:
            ds = ds.cache(
                self.cfg["cache"] if isinstance(self.cfg["cache"], str) else None
            )
        ds = ds.batch(self.cfg["batch_size"], drop_remainder=self.cfg["drop_remainder"])
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def build(self):
        import glob

        tr = (
            sorted(glob.glob(self.cfg["train_glob"]))
            if isinstance(self.cfg["train_glob"], str)
            else list(self.cfg["train_glob"])
        )
        if not tr:
            raise FileNotFoundError(
                f"No TFRecord files matched: {self.cfg['train_glob']}"
            )
        va = (
            sorted(glob.glob(self.cfg["val_glob"]))
            if isinstance(self.cfg["val_glob"], str) and self.cfg["val_glob"]
            else self.cfg["val_glob"]
        )
        te = (
            sorted(glob.glob(self.cfg["test_glob"]))
            if isinstance(self.cfg["test_glob"], str) and self.cfg["test_glob"]
            else self.cfg["test_glob"]
        )
        return self._read(tr, True), self._read(va, False), self._read(te, False)


class ExistingDatasetBuilder(DatasetBuilder):
    def __init__(
        self, train_ds, val_ds=None, test_ds=None, assume_batched: bool = True
    ):
        self.train_ds, self.val_ds, self.test_ds = train_ds, val_ds, test_ds
        self.assume_batched = assume_batched

    def build(self):
        return self.train_ds, self.val_ds, self.test_ds

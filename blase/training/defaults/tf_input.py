from typing import Any, Dict, Iterable, Optional, Union
import glob

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
        feature_spec: Optional[Dict[str, Any]] = None,  # now Optional
        parse_fn=None,
        augment_fn=None,
        batch_size: int = 256,
        shuffle_buffer: int = 10_000,
        repeat_train: bool = True,
        compression: Optional[str] = None,  # None|""|"GZIP"
        cache: Union[bool, str] = False,
        snapshot_dir: Optional[str] = None,
        drop_remainder: bool = True,
        num_parallel_reads: Optional[int] = None,
        num_parallel_calls: Optional[int] = None,
        prefetch: bool = True,
    ):
        comp = (compression or "").upper()
        if comp not in {"", "GZIP"}:
            raise ValueError(f"Unsupported compression: {compression!r}")
        self.cfg = dict(
            train_glob=train_glob,
            val_glob=val_glob,
            test_glob=test_glob,
            feature_spec=feature_spec,
            parse_fn=parse_fn,
            augment_fn=augment_fn,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            repeat_train=repeat_train,
            compression=comp if comp else None,
            cache=cache,
            snapshot_dir=snapshot_dir,
            drop_remainder=drop_remainder,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            prefetch=prefetch,
        )

    def _glob_list(self, pat_or_list):
        if pat_or_list is None:
            return None
        if isinstance(pat_or_list, str):
            return sorted(glob.glob(pat_or_list))
        return list(pat_or_list)

    def _read(self, files, training: bool):
        if not files:
            return None

        npr = self.cfg["num_parallel_reads"] or AUTOTUNE
        npc = self.cfg["num_parallel_calls"] or AUTOTUNE

        ds = tf.data.TFRecordDataset(
            files,
            compression_type=self.cfg["compression"] or "",
            num_parallel_reads=npr,
        )

        # Only parse if a feature_spec is provided
        if self.cfg["feature_spec"] is not None:
            spec = self.cfg["feature_spec"]
            ds = ds.map(
                lambda ex: tf.io.parse_single_example(ex, spec), num_parallel_calls=npc
            )

        if self.cfg["parse_fn"]:
            ds = ds.map(self.cfg["parse_fn"], num_parallel_calls=npc)

        if training and self.cfg["augment_fn"]:
            ds = ds.map(self.cfg["augment_fn"], num_parallel_calls=npc)

        if training:
            ds = ds.shuffle(self.cfg["shuffle_buffer"], reshuffle_each_iteration=True)
            if self.cfg["repeat_train"]:
                ds = ds.repeat()

        if self.cfg["cache"]:
            ds = ds.cache(
                self.cfg["cache"] if isinstance(self.cfg["cache"], str) else None
            )

        if self.cfg["snapshot_dir"]:
            ds = ds.apply(tf.data.experimental.snapshot(self.cfg["snapshot_dir"]))

        ds = ds.batch(self.cfg["batch_size"], drop_remainder=self.cfg["drop_remainder"])
        if self.cfg["prefetch"]:
            ds = ds.prefetch(AUTOTUNE)
        return ds

    def build(self):
        tr = self._glob_list(self.cfg["train_glob"])
        if not tr:
            raise FileNotFoundError(
                f"No TFRecord files matched: {self.cfg['train_glob']}"
            )
        va = self._glob_list(self.cfg["val_glob"])
        te = self._glob_list(self.cfg["test_glob"])
        return self._read(tr, True), self._read(va, False), self._read(te, False)


class ExistingDatasetBuilder(DatasetBuilder):
    def __init__(
        self, train_ds, val_ds=None, test_ds=None, assume_batched: bool = True
    ):
        self.train_ds, self.val_ds, self.test_ds = train_ds, val_ds, test_ds
        self.assume_batched = assume_batched

    def build(self):
        return self.train_ds, self.val_ds, self.test_ds

from typing import Any, Dict, Sequence, Optional, Mapping
import contextlib
import tensorflow as tf

from blase.training.protocols import (
    StrategyAdapter,
    PrecisionPolicy,
    Compiler,
    Fitter,
    Evaluator,
    Exporter,
    CallbackFactory,
)

# ---- Strategy / Precision ---------------------------------------------------


class TfStrategy(StrategyAdapter):
    def __init__(self, name: Optional[str] = "mirrored"):
        self._name = name
        self._strategy = self._build(name)

    def _build(self, name: Optional[str]) -> Optional[tf.distribute.Strategy]:
        if name is None:
            return None
        if name == "mirrored":
            return tf.distribute.MirroredStrategy()
        if name == "multiworker":
            return tf.distribute.MultiWorkerMirroredStrategy()
        if name == "tpu":
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            return tf.distribute.TPUStrategy(resolver)
        raise ValueError(f"Unknown TF strategy {name}")

    @contextlib.contextmanager
    def scope(self):
        if self._strategy:
            with self._strategy.scope():
                yield
        else:
            yield

    def shard_dataset(self, ds: Any, is_train: bool) -> Any:
        return ds  # tf.data is strategy-aware when created under scope


class TfPrecision(PrecisionPolicy):
    def __init__(self, policy: Optional[str] = "mixed_float16"):
        self._policy = policy

    def apply(self) -> None:
        if self._policy:
            tf.keras.mixed_precision.set_global_policy(self._policy)


# ---- Compile / Fit / Eval / Export -----------------------------------------


class TfCompiler(Compiler):
    def compile(
        self,
        model: Any,
        optimizer: Any,
        loss: Any,
        metrics: Sequence[Any],
        lr_schedule: Optional[Any],
        jit: Optional[bool] = None,
    ) -> Any:
        if callable(optimizer):
            optimizer = optimizer()
        if lr_schedule is not None and callable(lr_schedule):
            # If user passed factory, resolve to schedule and assign into optimizer if supported
            sched = lr_schedule()
            try:
                optimizer.learning_rate = sched
            except Exception:
                pass
        if jit is True:
            tf.config.optimizer.set_jit(True)
        norm_metrics = []
        for m in metrics or []:
            if isinstance(m, str):
                norm_metrics.append(m)
            elif hasattr(m, "update_state"):  # already a Metric instance
                norm_metrics.append(m)
            elif callable(m):  # functional metric -> wrap
                norm_metrics.append(
                    tf.keras.metrics.MeanMetricWrapper(
                        m, name=getattr(m, "__name__", "metric")
                    )
                )
            else:
                raise TypeError(f"Unsupported metric: {m!r}")
        model.compile(optimizer=optimizer, loss=loss, metrics=norm_metrics)
        return model


class TfFitter(Fitter):
    def fit(
        self,
        model: Any,
        train: Any,
        val: Optional[Any],
        epochs: int,
        steps_per_epoch: Optional[int],
        validation_steps: Optional[int],
        class_weight: Optional[Mapping[Any, float]],
        callbacks: Sequence[Any],
    ) -> Dict[str, Any]:
        hist = model.fit(
            train,
            validation_data=val,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            class_weight=class_weight,
            callbacks=list(callbacks),
        )
        return {"history": getattr(hist, "history", {}), "model": model}


class TfEvaluator(Evaluator):
    def evaluate(self, model: Any, ds: Any, steps: Optional[int]) -> Dict[str, float]:
        vals = model.evaluate(ds, steps=steps, return_dict=True)
        return dict(vals)


class TfExporter(Exporter):
    def export(
        self,
        model: Any,
        kind: str,
        path: Optional[str],
        extra: Optional[Dict[str, Any]],
    ) -> str:
        if kind == "saved_model":
            out = path or "export/saved_model"
            tf.saved_model.save(model, out)
            return out
        if kind == "tflite":
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            if extra and extra.get("optimizations"):
                converter.optimizations = extra["optimizations"]
            tflite = converter.convert()
            out = path or "export/model.tflite"
            tf.io.gfile.makedirs(tf.io.gfile.dirname(out))
            with tf.io.gfile.GFile(out, "wb") as f:
                f.write(tflite)
            return out
        if kind == "onnx":
            raise NotImplementedError("ONNX export not wired by default")
        raise ValueError(f"Unknown export kind {kind}")


# ---- Callbacks --------------------------------------------------------------


class TfCallbackFactory(CallbackFactory):
    def __init__(
        self,
        *,
        checkpoint: Optional[Dict[str, Any]] = None,
        tensorboard: Optional[Dict[str, Any]] = None,
        early_stopping: Optional[Dict[str, Any]] = None,
        reduce_lr: Optional[Dict[str, Any]] = None,
        progbar: Optional[Dict[str, Any]] = None,
        custom: Optional[Sequence[Any]] = None,
    ):
        self.cfg = dict(
            checkpoint=checkpoint,
            tensorboard=tensorboard,
            early_stopping=early_stopping,
            reduce_lr=reduce_lr,
            progbar=progbar,
            custom=list(custom or []),
        )

    def build(self) -> Sequence[Any]:
        cbs = []
        ck = self.cfg["checkpoint"]
        if ck:
            filename = ck.get("filename", "weights.{epoch:02d}-{val_loss:.4f}.h5")
            save_weights_only = ck.get("save_weights_only", True)

            # enforce suffix when saving weights only
            if save_weights_only and not filename.endswith(".weights.h5"):
                if filename.endswith(".h5"):
                    filename = filename[:-3] + ".weights.h5"
                else:
                    filename = filename + ".weights.h5"

            cbs.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=tf.io.gfile.join(ck.get("dir", "ckpt"), filename),
                    monitor=ck.get("monitor", "val_loss"),
                    mode=ck.get("mode", "min"),
                    save_best_only=ck.get("save_best_only", True),
                    save_weights_only=save_weights_only,
                )
            )
        tb = self.cfg["tensorboard"]
        if tb:
            cbs.append(tf.keras.callbacks.TensorBoard(log_dir=tb.get("log_dir", "tb")))
        es = self.cfg["early_stopping"]
        if es:
            cbs.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=es.get("monitor", "val_loss"),
                    patience=es.get("patience", 8),
                    mode=es.get("mode", "min"),
                    restore_best_weights=es.get("restore_best_weights", True),
                )
            )
        rl = self.cfg["reduce_lr"]
        if rl:
            cbs.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=rl.get("monitor", "val_loss"),
                    factor=rl.get("factor", 0.1),
                    patience=rl.get("patience", 4),
                    mode=rl.get("mode", "min"),
                )
            )
        cbs.extend(self.cfg["custom"])
        return cbs

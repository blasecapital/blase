from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

from blase.training.protocols import (
    StrategyAdapter,
    PrecisionPolicy,
    Tracker,
    DatasetBuilder,
    ModelFactory,
    Compiler,
    CallbackFactory,
    Fitter,
)

from blase.training.defaults import registry as default_registry

Backend = Literal["tensorflow", "torch"]
ExportKind = Literal["saved_model", "tflite", "onnx"]
Split = Literal["train", "val", "test"]


class Train:
    def __init__(
        self,
        *,
        backend: Backend = "tensorflow",
        strategy: Optional[StrategyAdapter] = None,
        precision: Optional[PrecisionPolicy] = None,
        tracker: Optional[Tracker] = None,
        run_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.strategy = strategy or default_registry.strategy(backend)
        self.precision = precision or default_registry.precision(backend)
        self.tracker = tracker or default_registry.tracker(run_dir)
        self.run_dir = run_dir
        self.seed = seed

        # Internal state
        self._ds_builder: Optional[DatasetBuilder] = None
        self._model_factory: Optional[ModelFactory] = None
        self._model_kwargs: Dict[str, Any] = {}
        self._compiler: Optional[Compiler] = None
        self._fitter: Fitter = default_registry.fitter(backend)
        self._callbacks_factory: Optional[CallbackFactory] = None
        self._model: Optional[Any] = None
        self._compiled: bool = False

    # ---------- Data sources ----------
    def from_tfrecords(
        self,
        *,
        train_glob: str,
        val_glob: Optional[Union[str, Iterable[str]]] = None,
        test_glob: Optional[Union[str, Iterable[str]]] = None,
        feature_spec: Dict[str, Any],
        parse_fn: Optional[Callable[..., Any]] = None,
        augment_fn: Optional[Callable[..., Any]] = None,
        batch_size: int = 256,
        shuffle_buffer: int = 10_000,
        repeat_train: bool = True,
        compression: Optional[str] = None,
        cache: Union[bool, str] = False,
        snapshot_dir: Optional[str] = None,
        drop_remainder: bool = True,
    ) -> "Train":
        if not train_glob:
            raise ValueError("from_tfrecords: train_glob is empty.")
        if not isinstance(feature_spec, dict) or not feature_spec:
            raise ValueError("from_tfrecords: feature_spec must be a non-empty dict.")

        cfg = dict(
            train_glob=train_glob,
            val_glob=val_glob,
            test_glob=test_glob,
            feature_spec=feature_spec,
            parse_fn=parse_fn,
            augment_fn=augment_fn,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            repeat_train=repeat_train,
            compression=compression,
            cache=cache,
            snapshot_dir=snapshot_dir,
            drop_remainder=drop_remainder,
        )
        self._ds_builder = default_registry.tf_dataset_builder(**cfg)
        return self

    def from_numpy(
        self,
        *,
        x_train: Any,
        y_train: Optional[Any] = None,
        x_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        x_test: Optional[Any] = None,
        y_test: Optional[Any] = None,
        batch_size: int = 256,
        shuffle: bool = True,
        prefetch: bool = True,
        drop_remainder: bool = True,
    ) -> "Train": ...

    def from_dataset(
        self, *, train_ds, val_ds=None, test_ds=None, assume_batched=True
    ) -> "Train":
        self._ds_builder = default_registry.wrap_existing_dataset(
            train_ds, val_ds, test_ds, assume_batched
        )
        return self

    # ---------- Model ----------
    def configure_model(
        self,
        *,
        model_fn: ModelFactory,
        model_kwargs: Optional[Dict[str, Any]] = None,
        compiler: Optional[Compiler] = None,
    ) -> "Train":
        self._model_factory = model_fn
        self._model_kwargs = dict(model_kwargs or {})
        self._compiler = compiler or self._compiler  # allow override here
        return self

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self._model_factory:
            raise RuntimeError(
                "configure_model(...) must be called before compile/fit."
            )
        self.precision.apply()
        with self.strategy.scope():
            self._model = self._model_factory(**self._model_kwargs)

    # ---------- Compile ----------
    def compile(
        self,
        *,
        optimizer: Union[str, Callable[[], Any], Any],
        loss: Union[str, Callable[..., Any], Any],
        metrics: Optional[List[Union[str, Callable[..., Any], Any]]] = None,
        lr_schedule: Optional[Union[Callable[..., Any], Any]] = None,
        jit: Optional[bool] = None,
        compiler: Optional[Compiler] = None,  # optional override here too
    ) -> "Train":
        self._ensure_model()
        self._compiler = (
            compiler or self._compiler or default_registry.compiler(self.backend)
        )
        if self._compiler is None:
            raise RuntimeError(
                "No Compiler provided and no default available for backend."
            )
        self._compiler.compile(
            self._model, optimizer, loss, metrics or [], lr_schedule, jit=jit
        )
        self._compiled = True
        return self

    # ---------- Callbacks / logging ----------
    def configure_callbacks(self, **cfg) -> "Train":
        self._callbacks_factory = default_registry.callbacks(self.backend, **cfg)
        return self

    # ---------- Fit / eval / export ----------
    def fit(
        self,
        *,
        epochs: int,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        class_weight: Optional[Dict[Union[int, str], float]] = None,
    ) -> Dict[str, Any]:
        if not self._compiled:
            raise RuntimeError("compile(...) must be called before fit(...).")
        if not self._ds_builder:
            raise RuntimeError("from_* data source must be configured before fit(...).")
        train_ds, val_ds, _ = self._ds_builder.build()
        callbacks = self._callbacks_factory.build() if self._callbacks_factory else []
        return self._fitter.fit(
            self._model,
            train_ds,
            val_ds,
            epochs,
            steps_per_epoch,
            validation_steps,
            class_weight,
            callbacks,
        )

    def evaluate(
        self,
        *,
        split: Split = "test",
        steps: Optional[int] = None,
    ) -> Dict[str, float]: ...

    def predict(
        self,
        *,
        split: Split = "test",
        steps: Optional[int] = None,
        return_numpy: bool = True,
    ) -> Any: ...

    def export(
        self,
        *,
        kind: ExportKind = "saved_model",
        path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,  # e.g., TFLite optimizations
    ) -> str: ...

    # ---------- Introspection ----------
    def summary(self) -> str: ...
    def artifacts(self) -> Dict[str, Any]: ...
    def plan(self) -> List[Tuple[str, Dict[str, Any]]]: ...

    # ---------- Convenience ----------
    @staticmethod
    def simple_tfrecord_run(
        *,
        train_glob: Union[str, Iterable[str]],
        val_glob: Optional[Union[str, Iterable[str]]] = None,
        feature_spec: Dict[str, Any],
        model_fn: Callable[..., Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        epochs: int = 1,
        batch_size: int = 256,
        run_dir: Optional[str] = None,
        strategy: Optional[str] = None,
        precision: Optional[str] = None,
    ) -> Dict[str, Any]: ...

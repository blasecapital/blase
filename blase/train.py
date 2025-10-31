from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union
from pathlib import Path
import json
import glob
import sys
import os
import platform

from blase.training.protocols import (
    StrategyAdapter,
    PrecisionPolicy,
    # Tracker,
    DatasetBuilder,
    ModelFactory,
    Compiler,
    CallbackFactory,
    Fitter,
)

from blase.track import Track
from blase.tracking.snapshot import build_code_blob
from blase.utils.hashing import Hash

from blase.training.defaults import registry as default_registry


def _as_list(
    x: Union[str, Path, Iterable[Union[str, Path]], None],
) -> List[Union[str, Path]]:
    if x is None:
        return []
    if isinstance(x, (str, Path)):
        return [x]
    return list(x)


def _expand_glob_list(
    globs: Union[str, Path, Iterable[Union[str, Path]], None],
) -> List[Path]:
    """Expand strings/paths or glob patterns to a sorted unique list of existing Paths."""
    out: List[Path] = []
    for item in _as_list(globs):
        s = str(item)
        p = Path(s).expanduser()
        if any(ch in s for ch in "*?[]"):
            matches = [Path(m) for m in glob.glob(s, recursive=True)]
            out.extend(matches)
        else:
            out.append(p)
    # de-dup and keep only existing files
    uniq = {p.resolve() for p in out if p.exists()}
    return sorted(uniq)


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
        track: bool = True,
        run_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.strategy = strategy or default_registry.strategy(backend)
        self.precision = precision or default_registry.precision(backend)
        self.tracker = Track.get(track)
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
        self._evaluator = None
        self._exporter = None

    # ---------- Data sources ----------
    def from_tfrecords(
        self,
        *,
        train_glob: Union[str, Iterable[str]],
        val_glob: Optional[Union[str, Iterable[str]]] = None,
        test_glob: Optional[Union[str, Iterable[str]]] = None,
        feature_spec: Optional[Dict[str, Any]] = None,
        parse_fn: Optional[Callable[..., Any]] = None,
        augment_fn: Optional[Callable[..., Any]] = None,
        batch_size: int = 256,
        shuffle_buffer: int = 10_000,
        repeat_train: bool = True,
        compression: Optional[str] = None,  # None|""|"GZIP"
        cache: Union[bool, str] = False,
        snapshot_dir: Optional[str] = None,
        drop_remainder: bool = True,
        num_parallel_reads: Optional[int] = None,  # None → AUTOTUNE in builder
        num_parallel_calls: Optional[int] = None,  # None → AUTOTUNE in builder
        prefetch: bool = True,
    ) -> "Train":
        if not train_glob:
            raise ValueError("from_tfrecords: train_glob is empty.")
        if feature_spec is None and parse_fn is None:
            raise ValueError("Provide feature_spec or parse_fn.")

        comp = (compression or "").upper()
        if comp not in {"", "GZIP"}:
            raise ValueError(f"Unsupported compression: {compression!r}")

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
            compression=comp if comp else None,
            cache=cache,
            snapshot_dir=snapshot_dir,
            drop_remainder=drop_remainder,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            prefetch=prefetch,
        )
        self._ds_cfg = cfg
        self._ds_builder = default_registry.tf_dataset_builder(**cfg)

        if self.tracker:
            params = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
            for k in ("train_glob", "val_glob", "test_glob"):
                g = params.get(k)
                if g is not None and not isinstance(g, (list, tuple)):
                    params[k] = [g]

            stream = self.tracker.stream(
                "blase.Train.from_tfrecords", params, code_fn=self.from_tfrecords
            )

            if parse_fn is not None:
                try:
                    blob, meta = build_code_blob(parse_fn)
                    parse_id = stream.step.register_data(
                        kind="code.json",
                        version="1",
                        path_or_bytes=blob,
                        metadata=meta,
                    )
                    stream.step.add_input(
                        parse_id, role="parse_fn", arg_name="parse_fn"
                    )
                except Exception:
                    pass

            if augment_fn is not None:
                try:
                    blob, meta = build_code_blob(augment_fn)
                    aug_id = stream.step.register_data(
                        kind="code.json", version="1", path_or_bytes=blob, metadata=meta
                    )
                    stream.step.add_input(
                        aug_id, role="augment_fn", arg_name="augment_fn"
                    )
                except Exception:
                    pass

            if feature_spec is not None:
                try:
                    fs_blob = json.dumps(feature_spec, sort_keys=True).encode("utf-8")
                    fs_id = stream.step.register_data(
                        kind="feature_spec.json",
                        version="1",
                        path_or_bytes=fs_blob,
                        metadata={},
                    )
                    stream.step.add_input(
                        fs_id, role="feature_spec", arg_name="feature_spec"
                    )
                except Exception:
                    pass

            for role, globs in (
                ("train", params["train_glob"]),
                ("val", params.get("val_glob") or []),
                ("test", params.get("test_glob") or []),
            ):
                for p in _expand_glob_list(globs):
                    try:
                        did = stream.step.register_data(
                            kind="tfrecord",
                            version="1",
                            path_or_bytes=str(p),
                            metadata={},
                        )
                        stream.step.add_input(did, role=role, arg_name=f"{role}_glob")
                    except Exception:
                        pass

            stream.emit(last_batch=True, meta={})
            stream.close_ok()

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

        if self.tracker:

            def _safe_cfg(d):
                try:
                    json.dumps(d, sort_keys=True)
                    return d
                except Exception:
                    return {"_hash": Hash().hash_object(d)}

            params = {
                "model_fn": getattr(
                    model_fn,
                    "__qualname__",
                    getattr(model_fn, "__name__", str(model_fn)),
                ),
                "model_kwargs": _safe_cfg(self._model_kwargs),
            }

            s = self.tracker.stream(
                "blase.Train.configure_model",
                params=params,
                code_fn=self.configure_model,
            )

            try:
                blob, meta = build_code_blob(model_fn)
                mid = s.step.register_data(
                    kind="code.json",
                    version="1",
                    path_or_bytes=blob,
                    metadata={},
                )
                s.step.add_input(
                    mid,
                    role="model_fn",
                    arg_name="model_fn",
                )
            except Exception:
                pass

            if compiler is not None:
                try:
                    blob, meta = build_code_blob(compiler.__class__)
                    cid = s.step.register_data(
                        kind="code.json",
                        version="1",
                        path_or_bytes=blob,
                        metadata=meta,
                    )
                    s.step.add_input(
                        cid,
                        role="compiler_cls",
                        arg_name=None,
                    )
                except Exception:
                    pass

            s.emit(last_batch=True, meta={})
            s.close_ok()

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

        if self.tracker:

            def _ser(obj):
                # Prefer get_config; fall back to fqname+repr
                try:
                    if hasattr(obj, "get_config"):
                        return {
                            "type": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                            "config": obj.get_config(),
                        }
                except Exception:
                    pass
                t = getattr(obj, "__class__", type(obj))
                return {
                    "type": f"{t.__module__}.{getattr(t, '__qualname__', t.__name__)}",
                    "repr": repr(obj),
                }

            params = {
                "optimizer": _ser(optimizer),
                "loss": _ser(loss),
                "metrics": [_ser(m) for m in (metrics or [])],
                "lr_schedule": _ser(lr_schedule) if lr_schedule is not None else None,
                "jit": jit,
            }

            s = self.tracker.stream(
                "blase.Train.compile",
                params=params,
                code_fn=self.compile,
            )

            # Snapshot custom callables as inputs
            def _maybe_snap(obj, role):
                if obj is None:
                    return
                try:
                    blob, meta = build_code_blob(obj)
                    did = s.step.register_data(
                        kind="code.json", version="1", path_or_bytes=blob, metadata=meta
                    )
                    s.step.add_input(did, role=role, arg_name=role)
                except Exception:
                    pass

            if callable(optimizer):
                _maybe_snap(optimizer, "optimizer")
            if callable(loss):
                _maybe_snap(loss, "loss")
            for m in metrics or []:
                if callable(m):
                    _maybe_snap(m, "metric")
            if callable(lr_schedule):
                _maybe_snap(lr_schedule, "lr_schedule")

            for lyr in getattr(self._model, "layers", []):
                mod = getattr(type(lyr), "__module__", "")
                if not mod.startswith(("tensorflow", "keras")):
                    _maybe_snap(type(lyr), "custom_layer")

            s.emit(last_batch=True, meta={})
            s.close_ok()

        return self

    # ---------- Callbacks / logging ----------
    def configure_callbacks(self, **cfg) -> "Train":
        self._callbacks_factory = default_registry.callbacks(self.backend, **cfg)

        if self.tracker:

            def _norm(v):
                if isinstance(v, Path):
                    return str(v)
                try:
                    json.dumps(v, sort_keys=True)
                    return v
                except Exception:
                    return {"_hash": Hash().hash_object(v)}

            params = {k: _norm(v) for k, v in cfg.items()}
            s = self.tracker.stream(
                "blase.Train.configure_callbacks",
                params=params,
                code_fn=self.configure_callbacks,
            )

            def _iter_objs(x):
                if x is None:
                    return
                if isinstance(x, (list, tuple, set)):
                    for i in x:
                        yield i
                else:
                    yield x

            for v in cfg.values():
                for obj in _iter_objs(v):
                    if callable(obj):
                        try:
                            blob, meta = build_code_blob(obj)
                            did = s.step.register_data(
                                kind="code.json",
                                version="1",
                                path_or_bytes=blob,
                                metadata=meta,
                            )
                            s.step.add_input(
                                did,
                                role="callback",
                                arg_name=None,
                            )
                        except Exception:
                            pass

            s.emit(last_batch=True, meta={})
            s.close_ok()

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

        if not self.tracker:
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

        def _safe_json(obj):
            try:
                json.dumps(obj, sort_keys=True)
                return obj
            except Exception:
                return {"_hash": Hash().hash_object(obj)}

        spec = {
            "backend": self.backend,
            "seed": self.seed,
            "data_cfg": _safe_json(getattr(self, "_ds_cfg", {})),
            "model_factory": getattr(
                self._model_factory, "__qualname__", str(self._model_factory)
            ),
            "model_kwargs": _safe_json(self._model_kwargs),
            "fit": {
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "validation_steps": validation_steps,
                "class_weight": _safe_json(class_weight),
            },
            "env": _safe_json(
                {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    # "tensorflow": getattr(tf, "__version__", None),
                    # "keras": getattr(tf.keras, "__version__", None) if hasattr(tf, "keras") else None,
                    # "numpy": getattr(np, "__version__", None),
                    "cuda": os.environ.get("CUDA_VERSION"),
                    "cudnn": os.environ.get("CUDNN_VERSION"),
                    # "devices": [d.name for d in tf.config.list_logical_devices()] if "tf" in globals() else None,
                }
            ),
        }

        stream = self.tracker.stream(
            "blase.Train.fit",
            params=spec,
            code_fn=self.fit,
        )

        def _snap(obj, role):
            try:
                b, m = build_code_blob(obj)
                did = stream.step.register_data("code.json","1", b, m)
                stream.step.add_input(did, role=role, arg_name=role)
            except Exception:
                pass

        # model factory already handled; also add loss/optimizer/metrics if custom
        comp = getattr(self, "_compiler", None)
        if comp is not None:
            for obj, role in [
                (getattr(comp, "optimizer", None), "optimizer"),
                (getattr(comp, "loss", None), "loss"),
                *[(m, "metric") for m in getattr(comp, "metrics", [])],
                (getattr(comp, "lr_schedule", None), "lr_schedule"),
            ]:
                if callable(obj):
                    _snap(obj, role)

        try:
            if self._model_factory is not None:
                blob, meta = build_code_blob(self._model_factory)
                mid = stream.step.register_data(
                    kind="code.json",
                    version="1",
                    path_or_bytes=blob,
                    metadata=meta,
                )
                stream.step.add_input(mid, role="model_fn", arg_name="model_fn")
        except Exception:
            pass

        ds_cfg = getattr(self, "_ds_cfg", {})
        for role, globs in (
            ("train", ds_cfg.get("train_glob")),
            ("val", ds_cfg.get("val_glob")),
            ("test", ds_cfg.get("test_glob")),
        ):
            for p in _expand_glob_list(globs):
                try:
                    did = stream.step.register_data(
                        kind="tfrecord",
                        version="1",
                        path_or_bytes=str(p),
                        metadata={},
                    )
                    stream.step.add_input(
                        data_hash=did,
                        role=role,
                        arg_name=f"{role}_glob",
                    )
                except Exception:
                    pass

        try:
            history = self._fitter.fit(
                self._model,
                train_ds,
                val_ds,
                epochs,
                steps_per_epoch,
                validation_steps,
                class_weight,
                callbacks,
            )

            try:
                final = {
                    k: (v[-1] if isinstance(v, (list, tuple)) and v else v)
                    for k, v in getattr(history, "history", {}).items()
                }
                hist_blob = json.dumps(
                    {
                        "keys": list(getattr(history, "history", {}).keys()),
                        "final": final,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            except Exception:
                # worst-case fallback
                hist_blob = json.dumps({"_repr": repr(history)}).encode("utf-8")

            hid = stream.step.register_data(
                kind="metrics.json", version="1", path_or_bytes=hist_blob, metadata={}
            )
            stream.step.add_output(hid, name="train_history")

            stream.emit(last_batch=True, meta={})
            stream.close_ok()
            return history

        except Exception as e:
            stream.close_error(type(e), e, e.__traceback__)
            raise

    def evaluate(self, *, split="test", steps=None):
        if self._model is None:
            raise RuntimeError("No model to evaluate. Call compile()/fit() first.")
        if not self._ds_builder:
            raise RuntimeError("No dataset. Call from_*() first.")
        ds = self._ds_builder.build()[{"train": 0, "val": 1, "test": 2}[split]]
        ev = self._evaluator or default_registry.evaluator(self.backend)
        return ev.evaluate(self._model, ds, steps)

    def predict(
        self,
        *,
        split: Split = "test",
        steps: Optional[int] = None,
        return_numpy: bool = True,
    ) -> Any:
        ds = self._ds_builder.build()[{"train": 0, "val": 1, "test": 2}[split]]
        ds_x = ds.map(lambda x, y: x)
        return self._model.predict(ds_x, steps=steps)

    def export(self, *, kind="saved_model", path=None, extra=None):
        if self._model is None:
            raise RuntimeError("No model to export. Call compile()/fit() first.")
        ex = self._exporter or default_registry.exporter(self.backend)
        return ex.export(self._model, kind, path, extra)

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

from typing import (
    Iterable,
    Optional,
    Dict,
    Any,
    Sequence,
    Mapping,
    Literal,
    Protocol,
    Union,
    List,
)
from pathlib import Path
from dataclasses import dataclass, field
import json

from blase.types import Batch, SinkResult
from blase.utils.hashing import Hash
from blase.track import Track

from blase.preparing.registry import PrepareRegistry

from blase.preparing.default_registry import default_registry

from blase.preparing.interfaces import ManifestBatch

from blase.preparing.manifest.manifest import build_manifest_stream
from blase.preparing.manifest import classmap as _classmap

from blase.preparing.stats import counters as _counters

from blase.preparing.splitters import random as _split_random
from blase.preparing.splitters import stratified as _split_strat
from blase.preparing.splitters import group as _split_group
from blase.preparing.splitters import time as _split_time

from blase.preparing.writers.tfrecord import writer as _tfr_writer
from blase.preparing.writers.tfrecord.inspect import head as _tfr_head

from blase.preparing.writers.sidecar import jsonl as _sidecar_jsonl
from blase.preparing.writers.sidecar import parquet as _sidecar_parquet

from blase.restoring import cas


# -------------------------
# Prepare: shared type hints
# -------------------------

ManifestRow = Dict[str, Any]
ManifestMeta = Dict[str, Any]
Splits = Mapping[str, Sequence[str]]
Stats = Dict[str, Any]
ClassMap = Mapping[str, int]


# -----------------------
# Plugin provider Protocol
# -----------------------


class ImageIndexProvider(Protocol):
    """Build a per-image index from parquet/arrow sources."""

    def build_index(
        self, sources: Sequence["DataSource"], cfg: "ImageConfig"
    ) -> Mapping[str, Dict[str, Any]]: ...


class LabelReader(Protocol):
    """Yield normalized label rows keyed by image_id."""

    def read(self, src: "LabelSource") -> Iterable[Dict[str, Any]]: ...


# -----------------------------
# Compact configuration objects
# -----------------------------


@dataclass(frozen=True)
class ImageConfig:
    id_col: str = "image_id"
    path_col: str = "path"
    bytes_col: Optional[str] = "image_bytes"
    height_col: str = "height"
    width_col: str = "width"
    sha256_col: Optional[str] = "sha256"


@dataclass(frozen=True)
class JoinConfig:
    image_id_resolver: Literal["provided", "sha256", "path_stem", "custom"] = "provided"
    custom_id_fn_fqn: Optional[str] = None
    join_on: str = "image_id"
    drop_orphans: bool = True
    keep_unlabeled_images: bool = True
    dedupe_policy: Literal["merge", "first", "last", "error"] = "merge"


@dataclass(frozen=True)
class BoxConfig:
    coord_in: Literal["xyxy_abs", "xywh_abs", "xyxy_rel", "xywh_rel"] = "xyxy_abs"
    coord_out: Literal["xyxy_rel", "xywh_rel"] = "xyxy_rel"
    clamp_boxes: bool = True
    drop_oob_boxes: bool = False


@dataclass(frozen=True)
class ClassConfig:
    class_map: Optional[ClassMap] = None
    normalize_names: bool = True


@dataclass(frozen=True)
class ScaleConfig:
    batch_rows: int = 4096
    seed: int = 42
    join_strategy: Literal["in_memory", "sqlite_temp", "duckdb_temp"] = "duckdb_temp"
    max_mem_mb: int = 2048
    index_backend: Literal["duckdb", "sqlite", "kv_parquet"] = "duckdb"
    spill_dir: Optional[Path] = None
    rows_per_chunk: int = 100_000
    bloom_on: Sequence[str] = ("image_id",)


# -------------------
# Source Descriptors
# -------------------


@dataclass(frozen=True)
class DataSource:
    uri: str
    fmt: Literal["parquet", "arrow"] = "parquet"
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelSource:
    kind: Literal["detection", "classification"]
    uri: str
    fmt: str  # "coco","jsonl_boxes","csv","parquet","yolo","voc", "labelbox"
    options: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------
# Conversion / Builder helpers
# ---------------------------


def images_parquet(uri: str, **opts: Any) -> DataSource:
    return DataSource(uri=uri, fmt="parquet", options=dict(opts))


def images_arrow(uri: str, **opts: Any) -> DataSource:
    return DataSource(uri=uri, fmt="arrow", options=dict(opts))


def coco(uri: str, **opts: Any) -> LabelSource:
    return LabelSource(kind="detection", uri=uri, fmt="coco", options=dict(opts))


def jsonl_boxes(uri: str, **opts: Any) -> LabelSource:
    return LabelSource(kind="detection", uri=uri, fmt="jsonl_boxes", options=dict(opts))


def csv_cls(uri: str, **opts: Any) -> LabelSource:
    return LabelSource(kind="classification", uri=uri, fmt="csv", options=dict(opts))


def parquet_det(uri: str, **opts: Any) -> LabelSource:
    return LabelSource(kind="detection", uri=uri, fmt="parquet", options=dict(opts))


def parquet_cls(uri: str, **opts: Any) -> LabelSource:
    return LabelSource(
        kind="classification", uri=uri, fmt="parquet", options=dict(opts)
    )


def _coerce_data_source(x: Union[DataSource, Mapping[str, Any]]) -> DataSource:
    if isinstance(x, DataSource):
        return x
    d = dict(x)
    return DataSource(
        uri=d["uri"], fmt=d.get("fmt", "parquet"), options=d.get("options", {}) or {}
    )


def _coerce_label_source(x: Union[LabelSource, Mapping[str, Any]]) -> LabelSource:
    if isinstance(x, LabelSource):
        return x
    d = dict(x)
    return LabelSource(
        kind=d["kind"], uri=d["uri"], fmt=d["fmt"], options=d.get("options", {}) or {}
    )


def _coerce_list_data_sources(
    seq: Optional[Sequence[Union[DataSource, Mapping[str, Any]]]],
) -> List[DataSource]:
    return [] if not seq else [_coerce_data_source(s) for s in seq]


def _coerce_list_label_sources(
    seq: Optional[Sequence[Union[LabelSource, Mapping[str, Any]]]],
) -> List[LabelSource]:
    return [] if not seq else [_coerce_label_source(s) for s in seq]


# -----------------------
# Presets
# -----------------------


class PreparePreset:
    @staticmethod
    def coco_parquet(
        images_glob: str,
        coco_json: str,
        image_cfg: ImageConfig = ImageConfig(),
        join_cfg: JoinConfig = JoinConfig(image_id_resolver="provided"),
        box_cfg: BoxConfig = BoxConfig(),
        class_cfg: ClassConfig = ClassConfig(),
        scale_cfg: ScaleConfig = ScaleConfig(join_strategy="duckdb_temp"),
    ):
        ds = [images_parquet(images_glob)]
        ls = [coco(coco_json)]
        return ds, ls, image_cfg, join_cfg, box_cfg, class_cfg, scale_cfg


class Prepare:
    """
    Prepares cleaned and aligned datasets for training by applying splits, shaping,
    and converting them into optimized binary formats compatible with major ML frameworks.

    The `Prepare` module serves as the final transformation stage before training.
    It coordinates multi-source alignment (e.g., features, targets, metadata), applies
    dataset splits, and writes training-ready files in efficient I/O formats such as
    `.npy`, `.npz`, `.tfrecord`, and others.

    It is designed to be flexible across data types—including tabular, time series,
    text, image, audio, and reinforcement learning inputs—and can integrate outputs
    from the `Extract`, `Transform`, and `Clean` modules. For time-dependent or
    episodic data (e.g., RL or forecasting), it supports rolling window generation
    and grouped sampling.

    Key Responsibilities:
    ---------------------
    - Register and align multiple data sources using shared keys or indices.
    - Apply train/val/test splits with optional stratification or reproducible seeding.
    - Convert data into binary formats optimized for TensorFlow, PyTorch, and XGBoost.
    - Handle exclusion lists (from `Clean`) to remove invalid samples pre-conversion.
    - Support windowed or episodic dataset shaping for RL and time series tasks.
    - Log output metadata for reproducibility (e.g., sample counts, shape, exclusions, splits).

    Methods:
    --------
    register_source(name: str, data: Iterable)
        Register a named data source (e.g., "features", "targets", "metadata").

    load_exclusion_keys(tag: str, path: Optional[str] = None)
        Load a list of keys to exclude (generated by the `Clean` module).

    set_split(train: float, val: float = 0.0, test: float = 0.0, seed: Optional[int] = None)
        Define split ratios and optional random seed for reproducibility.

    apply_rolling_windows(window_size: int, step: int = 1)
        Apply sliding window reshaping to time series or RL-style input data.

    convert(output_format: str = "npy", output_dir: str = "./data/processed/")
        Convert registered sources into aligned, formatted training datasets.

    prepare_rl_dataset(state_source, action_source, reward_source, next_state_source, done_source, ...)
        Align and format structured RL logs into (state, action, reward, next_state, done) transitions.

    save_metadata(path: str = "./data/processed/metadata.json")
        Store conversion configuration and metadata for reproducibility.

    Example:
    --------
    >>> prepare = Prepare()
    >>> prepare.register_source("features", extractor.load_csv("features.csv", batch_size=1000))
    >>> prepare.register_source("targets", extractor.load_csv("targets.csv", batch_size=1000))
    >>> prepare.load_exclusion_keys("cleaned_v1")
    >>> prepare.set_split(train=0.7, val=0.2, test=0.1, seed=42)
    >>> prepare.apply_rolling_windows(window_size=30)
    >>> prepare.convert(output_format="npy", output_dir="./data/prepared/")

    Notes:
    ------
    - All registered sources must align on sample keys or row order.
    - Data is written to disk in a format optimized for your selected ML framework.
    - For RL workflows, structured logs can be converted to agent-ready transitions.
    - K-fold splitting and online replay buffers are not handled here—they belong in `Train`.
    """

    def __init__(self, registry: Union[PrepareRegistry, None] = None):
        self._reg = registry or default_registry()
        self._stream = None
        self._manifest_id = None

    def build_manifest(
        self,
        *,
        data_sources=None,
        label_sources=None,
        image_cfg=ImageConfig(),
        join_cfg=JoinConfig(),
        box_cfg=BoxConfig(),
        class_cfg=ClassConfig(),
        scale_cfg=ScaleConfig(),
        track: bool = True,
    ) -> Iterable[ManifestBatch]:
        tracker = Track.get(track)

        # coerce + normalize
        ds = _coerce_list_data_sources(data_sources)
        ls = _coerce_list_label_sources(label_sources)
        ds_norm = [{"uri": d.uri, "fmt": d.fmt, "options": dict(d.options)} for d in ds]
        ls_norm = [
            {"kind": s.kind, "uri": s.uri, "fmt": s.fmt, "options": dict(s.options)}
            for s in ls
        ]

        # flattened param view (serializable)
        params = {
            "data_sources": ds_norm,
            "label_sources": ls_norm,
            "image_cfg": {
                "id_col": image_cfg.id_col,
                "path_col": image_cfg.path_col,
                "height_col": image_cfg.height_col,
                "width_col": image_cfg.width_col,
                "sha256_col": image_cfg.sha256_col,
            },
            "join_cfg": {
                "image_id_resolver": join_cfg.image_id_resolver,
                "drop_orphans": join_cfg.drop_orphans,
                "keep_unlabeled_images": join_cfg.keep_unlabeled_images,
                "dedupe_policy": join_cfg.dedupe_policy,
            },
            "box_cfg": {
                "coord_in": box_cfg.coord_in,
                "coord_out": box_cfg.coord_out,
                "clamp_boxes": box_cfg.clamp_boxes,
                "drop_oob_boxes": box_cfg.drop_oob_boxes,
            },
            "class_cfg": {
                "class_map": class_cfg.class_map,
                "normalize_names": class_cfg.normalize_names,
            },
            "scale_cfg": {
                "batch_rows": scale_cfg.batch_rows,
                "seed": scale_cfg.seed,
                "join_strategy": scale_cfg.join_strategy,
                "rows_per_chunk": scale_cfg.rows_per_chunk,
                "index_backend": scale_cfg.index_backend,
            },
        }

        # ============== untracked ==============
        if tracker is None:
            return build_manifest_stream(reg=self._reg, **params)

        # ============== tracked ==============
        def _tracked_iter():
            new_stream = (
                self._stream is None or getattr(self._stream, "params", None) != params
            )
            if new_stream and self._stream is not None:
                try:
                    self._stream.close_ok()
                except Exception:
                    pass

            if new_stream:
                self._stream = tracker.stream(
                    "blase.Prepare.build_manifest",
                    params,
                    code_fn=self.build_manifest,
                )
                setattr(self._stream, "params", params)

                # register as an output once; use name 'manifest'
                try:
                    data_id = self._stream.step.register_data(
                        kind="manifest",
                        version="1",
                        path_or_bytes=b"",
                        metadata={"params_hash": Hash().hash_object(params)},
                    )
                    self._stream.step.add_output(data_id, name="manifest")
                    self._manifest_id = data_id
                except Exception:
                    pass

            try:
                inner = build_manifest_stream(reg=self._reg, **params)
                for mb in inner:
                    # enrich meta with manifest ids
                    meta = dict(mb.meta or {})
                    meta.setdefault("manifest_root_hash", self._manifest_id)
                    meta.setdefault("manifest_hash", self._manifest_id)

                    new_meta = self._stream.emit(last_batch=mb.is_last, meta=meta)
                    yield Batch(
                        data=mb.data,
                        meta=(new_meta or meta),
                        is_last=mb.is_last,
                    )

                self._stream.close_ok()
                self._stream = None
                self._manifest_id = None
            except Exception as e:
                try:
                    self._stream.close_error(type(e), e, e.__traceback__)
                finally:
                    self._stream = None
                    self._manifest_id = None
                raise

        return _tracked_iter()

    def compute_stats(
        self,
        manifest: Iterable[ManifestBatch],
        *,
        by: Literal["class", "image", "global"] = "class",
        track: bool = True,
    ) -> Batch[Stats, ManifestMeta]:
        tracker = Track.get(track)
        if tracker is None:
            s = _counters.compute(manifest, {"by": by})
            return Batch(data=s, meta={"by": by}, is_last=True)

        step = tracker.stream(
            "blase.Prepare.compute_stats", {"by": by}, code_fn=self.compute_stats
        )
        # plumb upstream manifest hashes into the step
        try:
            # one pass compute while capturing upstream meta
            upstream = {"manifest_root_hash": None, "manifest_hash": None}

            def _iter():
                for mb in manifest:
                    rh = (mb.meta or {}).get("manifest_root_hash")
                    mh = (mb.meta or {}).get("manifest_hash")
                    upstream["manifest_root_hash"] = (
                        upstream["manifest_root_hash"] or rh
                    )
                    upstream["manifest_hash"] = upstream["manifest_hash"] or mh
                    yield mb

            stats = _counters.compute(_iter(), {"by": by})
            # add inputs if available
            for rid, role in [
                (upstream["manifest_root_hash"], "manifest_root"),
                (upstream["manifest_hash"], "manifest"),
            ]:
                if rid:
                    try:
                        step.step.add_input(rid, role=role, arg_name=None)
                    except Exception:
                        pass
            meta = {"by": by}
            new_meta = step.emit(last_batch=True, meta=meta) or meta
            step.close_ok()
            return Batch(data=stats, meta=new_meta, is_last=True)
        except Exception as e:
            step.close_error(type(e), e, e.__traceback__)
            raise

    def split(
        self,
        manifest: Iterable[ManifestBatch],
        *,
        method: Literal["random", "stratified", "group", "time"] = "stratified",
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int = 42,
        group_key: str = "image_id",
        stratify_on: str = "class",
        holdout_query: Optional[str] = None,
        track: bool = True,
    ) -> Batch[Splits, ManifestMeta]:
        cfg = {
            "fractions": {"train": train, "val": val, "test": test},
            "seed": seed,
            "group_key": group_key,
            "stratify_on": stratify_on,
            "holdout_query": holdout_query,
        }
        if abs(train + val + test - 1.0) > 1e-6:
            raise ValueError("train+val+test must equal 1.0")

        # choose splitter
        if method == "random":
            do_split = _split_random.split
        elif method == "stratified":
            do_split = _split_strat.split
        elif method == "group":
            do_split = _split_group.split
        elif method == "time":
            do_split = _split_time.split
        else:
            raise ValueError(f"unknown split method {method!r}")

        tracker = Track.get(track)
        if tracker is None:
            s = do_split(manifest, cfg)
            meta: ManifestMeta = {
                "method": method,
                "fractions": cfg["fractions"],
                "seed": seed,
                "sizes": {k: len(v) for k, v in s.items()},
            }
            return Batch(data=s, meta=meta, is_last=True)

        # tracked path
        upstream = {"root": None, "hash": None}

        def _iter_capture():
            for mb in manifest:
                rh = (mb.meta or {}).get("manifest_root_hash")
                mh = (mb.meta or {}).get("manifest_hash")
                if rh and not upstream["root"]:
                    upstream["root"] = rh
                if mh and not upstream["hash"]:
                    upstream["hash"] = mh
                yield mb

        params = {
            "method": method,
            "fractions": cfg["fractions"],
            "seed": seed,
            "group_key": group_key,
            "stratify_on": stratify_on,
            "holdout_query": holdout_query,
        }
        step = tracker.stream("blase.Prepare.split", params, code_fn=self.split)

        try:
            s = do_split(_iter_capture(), cfg)

            # record upstream manifest identity if available
            for rid, role in (
                (upstream["root"], "manifest_root"),
                (upstream["hash"], "manifest"),
            ):
                if rid:
                    try:
                        step.step.add_input(rid, role=role, arg_name=None)
                    except Exception:
                        pass

            # register a splits artifact id (metadata only to avoid large payloads)
            sizes = {k: len(v) for k, v in s.items()}
            splits_meta = {
                "method": method,
                "fractions": cfg["fractions"],
                "seed": seed,
                "sizes": sizes,
            }
            try:
                splits_id = step.step.register_data(
                    kind="splits", version="1", path_or_bytes=b"", metadata=splits_meta
                )
                step.step.add_output(splits_id, name="splits")
            except Exception:
                splits_id = None

            meta: ManifestMeta = {
                **splits_meta,
                "upstream_manifest_root": upstream["root"],
                "upstream_manifest": upstream["hash"],
                "splits_id": splits_id,
            }
            new_meta = step.emit(last_batch=True, meta=meta) or meta
            step.close_ok()
            return Batch(data=s, meta=new_meta, is_last=True)
        except Exception as e:
            step.close_error(type(e), e, e.__traceback__)
            raise

    def to_tfrecord(
        self,
        manifest: Iterable[ManifestBatch],
        splits: Splits,
        *,
        out_dir: Path,
        mode: Literal["combined", "images_labels"] = "combined",
        shard_size_mb: int = 128,
        compression: Literal["GZIP", "NONE"] = "GZIP",
        include_image_bytes: bool = True,
        read_bytes_from: Literal["parquet", "filesystem"] = "parquet",
        parquet_id_col="image_id",
        parquet_bytes_col="img_bytes",
        deterministic_order: bool = True,
        order_key: Literal["sha256", "image_id"] = "sha256",
        write_workers: Optional[int] = None,
        write_alignment_index: bool = True,
        example_id_feature: str = "example/id",
        track: bool = True,
    ) -> SinkResult:
        cfg = {
            "out_dir": Path(out_dir),
            "mode": mode,
            "shard_size_mb": int(shard_size_mb),
            "compression": compression,
            "include_image_bytes": bool(include_image_bytes),
            "read_bytes_from": read_bytes_from,
            "example_id_feature": example_id_feature,
            "parquet_id_col": parquet_id_col,
            "parquet_bytes_col": parquet_bytes_col,
            "deterministic_order": bool(deterministic_order),
            "order_key": order_key,
            "write_workers": write_workers,
            "write_alignment_index": bool(write_alignment_index),
        }

        tracker = Track.get(track)
        if tracker is None:
            return _tfr_writer.write(manifest, splits, cfg)

        # ---------- tracked path ----------
        # 1) params recorded on the step (stringify paths for stability)
        params = {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()},
            "out_dir": str(cfg["out_dir"]),
        }
        stream = tracker.stream(
            "blase.Prepare.to_tfrecord", params, code_fn=self.to_tfrecord
        )

        # 2) capture upstream manifest ids while passing through
        it = iter(manifest)
        first_mb = None
        try:
            first_mb = next(it)
        except StopIteration:
            first_mb = None

        upstream = {"manifest_root_hash": None, "manifest_hash": None}
        if first_mb is not None and isinstance(first_mb.meta, dict):
            upstream["manifest_root_hash"] = first_mb.meta.get("manifest_root_hash")
            upstream["manifest_hash"] = first_mb.meta.get("manifest_hash")

        # 3) register splits blob as a first-class input
        try:
            splits_json = json.dumps(splits, sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        except Exception:
            # best effort; fall back to hashing the python object
            splits_json = json.dumps({"_hash": Hash().hash_object(splits)}).encode(
                "utf-8"
            )

        try:
            splits_hash = stream.step.register_data(
                kind="splits.json", version="1", path_or_bytes=splits_json, metadata={}
            )
            try:
                cas.write_bytes(
                    run_path=stream.step.run_path,
                    kind="data",
                    data_hash=splits_hash,
                    data=splits_json,
                )
            except Exception:
                pass
            stream.step.add_input(splits_hash, role="splits", arg_name="splits")
        except Exception:
            splits_hash = None  # non-fatal

        # 4) add manifest inputs if available
        for rid, role in [
            (upstream["manifest_root_hash"], "manifest_root"),
            (upstream["manifest_hash"], "manifest"),
        ]:
            if rid:
                try:
                    stream.step.add_input(rid, role=role, arg_name=None)
                except Exception:
                    pass

        try:
            # 5) delegate write; collect artifacts
            def _manifest_iter_for_writer():
                if first_mb is not None:
                    yield first_mb
                for mb in it:
                    yield mb

            res = _tfr_writer.write(_manifest_iter_for_writer(), splits, cfg)

            # 6) register each produced file as outputs
            artifacts = res.artifacts or []
            for i, a in enumerate(artifacts, 1):
                kind = (
                    "tfrecord.index"
                    if ("index" in (a.kind or "").lower() or a.path.endswith(".index"))
                    else (a.kind or "tfrecord")
                )
                try:
                    data_id = stream.step.register_data(
                        kind=kind,
                        version="1",
                        path_or_bytes=a.path,
                        metadata={"path": a.path},
                    )
                    stream.step.add_output(data_id, name=f"{kind}_{i:05d}")
                except Exception:
                    pass

            # 7) emit final meta + close
            meta = dict(res.meta or {})
            new_meta = stream.emit(last_batch=True, meta=meta) or meta
            stream.close_ok()
            return SinkResult(artifacts=artifacts, is_last=res.is_last, meta=new_meta)

        except Exception as e:
            stream.close_error(type(e), e, e.__traceback__)
            raise

    def preview_tfrecord(
        self,
        path: Path,
        *,
        n: int = 8,
        compression: str = "GZIP",
        decode_images: bool = False,
        out_dir: Optional[Path] = None,
        max_side: int = 1024,
    ) -> List[Dict[str, Any]]:
        return _tfr_head(
            path=Path(path),
            n=n,
            compression=compression,
            decode_images=decode_images,
            out_dir=out_dir,
            max_side=max_side,
        )

    def write_label_sidecars(
        self,
        manifest: Iterable[ManifestBatch],
        splits: Splits,
        *,
        out_dir: Path,
        format: Literal["jsonl", "parquet"] = "jsonl",
        deterministic_order: bool = True,
        order_key: Literal["sha256", "image_id"] = "sha256",
    ) -> SinkResult:
        cfg: Dict[str, Any] = {
            "out_dir": Path(out_dir),
            "deterministic_order": bool(deterministic_order),
            "order_key": order_key,
        }
        if format == "jsonl":
            return _sidecar_jsonl.write(manifest, splits, cfg)
        if format == "parquet":
            return _sidecar_parquet.write(manifest, splits, cfg)
        raise ValueError(f"unsupported sidecar format: {format!r}")

    def class_map_io(
        self,
        *,
        load: Optional[Path] = None,
        save: Optional[Path] = None,
        map: Optional[Dict[str, int]] = None,
        normalize: bool = True,
    ) -> Batch[Dict[str, int], ManifestMeta]:
        """
        Load/save a canonical class→id map.
        Priority: `map` arg > file at `load`. If `save` is provided, persist the final map.
        Returns Batch[data=<class_map>, meta={...}].
        """
        # 1) assemble source
        if map is not None:
            cm = dict(map)
        elif load is not None:
            p = Path(load)
            txt = p.read_text(encoding="utf-8")
            cm = json.loads(txt)
        else:
            cm = {}

        # 2) normalize + validate + canonicalize
        cm_norm = _classmap.canonicalize_class_map(cm, normalize_names=normalize)

        # 3) persist if requested
        if save is not None:
            sp = Path(save)
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text(
                json.dumps(cm_norm, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        # 4) meta
        j = json.dumps(cm_norm, sort_keys=True, ensure_ascii=False).encode("utf-8")
        meta: ManifestMeta = {
            "size": len(cm_norm),
            "normalized": bool(normalize),
            "hash": Hash().hash_object(j),
            "loaded_from": str(load) if load else None,
            "saved_to": str(save) if save else None,
        }
        return Batch(data=cm_norm, is_last=True, meta=meta)

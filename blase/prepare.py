from typing import (
    Iterable,
    Optional,
    Dict,
    Any,
    Sequence,
    Mapping,
    Literal,
    Iterator,
    Protocol,
    Union,
    List,
)
from pathlib import Path
from dataclasses import dataclass, field

from blase.types import Batch, SinkResult
from blase.preparing.registry import PrepareRegistry
from blase.preparing.default_registry import default_registry
from blase.preparing.interfaces import ManifestBatch
from blase.preparing.manifest import index_images as _idx
from blase.preparing.manifest import align_stream as _align
from blase.preparing.manifest import classmap as _classmap


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

    def build_manifest(
        self,
        *,
        data_sources: Optional[Sequence[Union[DataSource, Mapping[str, Any]]]] = None,
        label_sources: Optional[Sequence[Union[LabelSource, Mapping[str, Any]]]] = None,
        image_cfg: ImageConfig = ImageConfig(),
        join_cfg: JoinConfig = JoinConfig(),
        box_cfg: BoxConfig = BoxConfig(),
        class_cfg: ClassConfig = ClassConfig(),
        scale_cfg: ScaleConfig = ScaleConfig(),
    ) -> Iterator[ManifestBatch]:
        # coerce inputs
        ds = _coerce_list_data_sources(data_sources)
        ls = _coerce_list_label_sources(label_sources)

        # 1) images → KV index
        kv_index, bloom, rg_index = _idx.build_image_index(
            data_sources=[
                {"uri": d.uri, "fmt": d.fmt, "options": dict(d.options)} for d in ds
            ],
            image_cfg={
                "id_col": image_cfg.id_col,
                "path_col": image_cfg.path_col,
                "height_col": image_cfg.height_col,
                "width_col": image_cfg.width_col,
                "sha256_col": image_cfg.sha256_col,
            },
            scale_cfg={
                "rows_per_chunk": scale_cfg.rows_per_chunk,
                "index_backend": scale_cfg.index_backend,
            },
        )

        # 2) labels → iterables (via registry.label_readers)
        label_iters: list[Iterable[Dict[str, Any]]] = []
        for src in ls:
            reader = self._reg.label_readers.get(src.fmt)
            if reader is None:
                raise ValueError(f"no label reader registered for fmt={src.fmt!r}")
            label_iters.append(
                reader.read(
                    {"uri": src.uri, "fmt": src.fmt, "options": dict(src.options)}
                )
            )

        # 3) class map
        class_map, class_meta = _classmap.build_or_validate_class_map(
            label_iters,
            {
                "class_map": class_cfg.class_map,
                "normalize_names": class_cfg.normalize_names,
            },
        )

        # 4) align stream
        yield from _align.align_stream(
            label_iters=label_iters,
            kv_index=kv_index,
            rg_index=rg_index,
            join_cfg={
                "image_id_resolver": join_cfg.image_id_resolver,
                "drop_orphans": join_cfg.drop_orphans,
                "keep_unlabeled_images": join_cfg.keep_unlabeled_images,
                "dedupe_policy": join_cfg.dedupe_policy,
            },
            box_cfg={
                "coord_in": box_cfg.coord_in,
                "coord_out": box_cfg.coord_out,
                "clamp_boxes": box_cfg.clamp_boxes,
                "drop_oob_boxes": box_cfg.drop_oob_boxes,
            },
            class_cfg={
                "class_map": class_map,
                "normalize_names": class_cfg.normalize_names,
            },
            scale_cfg={
                "batch_rows": scale_cfg.batch_rows,
                "seed": scale_cfg.seed,
                "join_strategy": scale_cfg.join_strategy,
            },
        )

    def compute_stats(
        self,
        manifest: Iterable[ManifestBatch],
        *,
        by: Literal["class", "image", "global"] = "class",
    ) -> Batch[Stats, ManifestMeta]: ...

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
    ) -> Batch[Splits, ManifestMeta]: ...

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
        deterministic_order: bool = True,
        order_key: Literal["sha256", "image_id"] = "sha256",
        write_workers: Optional[int] = None,
        write_alignment_index: bool = True,
        example_id_feature: str = "example/id",
    ) -> SinkResult: ...

    def write_label_sidecars(
        self,
        manifest: Iterable[ManifestBatch],
        splits: Splits,
        *,
        out_dir: Path,
        format: Literal["jsonl", "parquet"] = "jsonl",
        deterministic_order: bool = True,
        order_key: Literal["sha256", "image_id"] = "sha256",
    ) -> SinkResult: ...

    def class_map_io(
        self,
        *,
        load: Optional[Path] = None,
        save: Optional[Path] = None,
        map: Optional[ClassMap] = None,
        normalize: bool = True,
    ) -> Batch[Dict[str, int], ManifestMeta]: ...

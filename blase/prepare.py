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
# Main object
# -----------------------


class Prepare:
    """
    High-level dataset preparation façade.

    This class wires together common “prepare” operations (manifest building,
    dataset splitting, TFRecord/sidecar generation, previews, and class-map I/O)
    with optional **tracking** so that every step can be deterministically
    **replayed** later from the run history.

    # What it does

    - build_manifest(...)
        Construct a streaming manifest from data + label sources with configurable
        joins, box transforms, class normalization, and batching. When tracked,
        the step records normalized params and emits stable `manifest_root` /
        `manifest` identities in batch metadata for downstream provenance.

    - split(manifest, ...)
        Produce train/val/test splits (random/stratified/group/time). When tracked,
        the step records upstream manifest identities as inputs and registers a
        lightweight `splits.json` blob (metadata-only unless you enable explicit
        CAS persistence elsewhere). Returns a `Batch` with `data=<Splits dict>`.

    - to_tfrecord(manifest, splits, ...)
        Write TFRecords (optionally with image bytes) with deterministic sharding,
        compression, and alignment index. When tracked, registers the `splits`
        blob as an input, links upstream manifest identities, and registers each
        produced artifact (`.tfrecord`, `.index`) as outputs.

    - preview_tfrecord(path, n=..., decode_images=..., out_dir=..., ...)
        Peek into a TFRecord and (optionally) write decoded thumbnails. When
        tracked, records the TFRecord as an input and each written thumbnail as
        an output. Also registers a small summary blob for quick inspection.

    - write_label_sidecars(manifest, splits, out_dir, format="jsonl"/"parquet", ...)
        Emit label “sidecar” files for each split with deterministic ordering.
        When tracked, links upstream manifest identities, records the `splits`
        blob, and registers each sidecar file as an output.

    - class_map_io(load=..., save=..., map=..., normalize=True)
        Load/save a canonical class→id map with normalization. When tracked,
        inline `map` is stored as a CAS blob input; `load` sources are recorded
        by content hash; an optional saved JSON file is registered as an output.

    # Tracking & Replay

    All methods accept `track: bool = True`. With tracking enabled:
      • Parameters are normalized and stored on the step.
      • Upstream data identities (e.g., manifest, splits, source files) are added
        as **inputs**.
      • Produced artifacts (files) are registered as **outputs**.
      • Batches the class emits carry manifest identity metadata so downstream
        steps can stitch lineage automatically.

    The CLI can then:
      • **Step replay**: `blase restore run --step <hash> --mode replay [--to <path>]`
        Reconstruct computation from recorded params/inputs; write artifacts to
        `--to` (file/dir) or to the original recorded locations.
      • **Data restore**: `blase restore run --data <data_hash> [--to <path>]`
        Materialize a single recorded artifact by its content hash; if missing
        locally, the producing step is replayed to regenerate it.

    # Return types

    - Streaming producers yield `ManifestBatch` / `Batch` objects.
    - Sinks return `SinkResult` capturing produced artifacts, `is_last`, and meta.
    - Utility methods like `class_map_io` return a `Batch[data=<payload>, meta=...]`.

    # Notes

    - Determinism: many writers accept `deterministic_order` and `order_key`
      (`"sha256"` / `"image_id"`) so that sink outputs are stable across runs.
    - Splits: if you do not persist `splits.json` separately, replay handlers will
      reconstruct it from the recorded input blob embedded on the `to_tfrecord` /
      sidecar steps.
    - Bytes: some methods may store small JSON blobs directly in CAS; large file
      payloads are referenced by path and verified by content hash during restore.
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
        """
        Build a streaming **manifest** of images and labels.

        The manifest is emitted as a generator of `ManifestBatch` chunks that
        downstream steps (e.g., `split`, `to_tfrecord`, sidecar writers) can
        consume without loading the entire dataset into memory. When tracking is
        enabled, this step records normalized parameters and emits stable
        manifest identities in batch metadata to enable deterministic replay.

        Parameters
        ----------
        data_sources : list or None, optional
            Image table sources. Each item should be created by a `prepare` source
            helper (e.g., `images_parquet(...)`) or be an object with the fields
            `uri`, `fmt`, and `options`. At runtime these are normalized to a list of
            dicts like:
            `{"uri": <str>, "fmt": <str>, "options": {…}}`.
            Defaults to `None`.
        label_sources : list or None, optional
            Label sources (e.g., COCO JSON, Labelbox NDJSON). Each item should be
            created by a `prepare` label helper (e.g., `coco(...)`, `labelbox(...)`)
            or be an object with fields `kind`, `uri`, `fmt`, and `options`.
            Normalized to:
            `{"kind": <str>, "uri": <str>, "fmt": <str>, "options": {…}}`.
            Defaults to `None`.
        image_cfg : ImageConfig, optional
            Column mapping for the image table. Relevant fields:
            `id_col`, `path_col`, `height_col`, `width_col`, `sha256_col`.
            Defaults to `ImageConfig()`.
        join_cfg : JoinConfig, optional
            Controls how image rows and labels are associated. Fields include:
            `image_id_resolver`, `drop_orphans`, `keep_unlabeled_images`,
            `dedupe_policy`. Defaults to `JoinConfig()`.
        box_cfg : BoxConfig, optional
            Bounding box conversion and validation. Fields:
            `coord_in`, `coord_out`, `clamp_boxes`, `drop_oob_boxes`.
            Defaults to `BoxConfig()`.
        class_cfg : ClassConfig, optional
            Class-name normalization and (optional) fixed class map. Fields:
            `class_map`, `normalize_names`. Defaults to `ClassConfig()`.
        scale_cfg : ScaleConfig, optional
            Streaming/batching and join backend controls. Fields:
            `batch_rows`, `seed`, `join_strategy`, `rows_per_chunk`, `index_backend`.
            Defaults to `ScaleConfig()`.
        track : bool, optional
            If `True`, record this step (parameters + lineage) and enrich emitted
            batch metadata with stable manifest identities for replay. Defaults to `True`.

        Yields
        ------
        ManifestBatch
            A batch object with attributes:
            - `data` : implementation-specific batch payload (e.g., a table/rows).
            - `meta` : dict containing, at minimum when tracked:
                * `"manifest_root_hash"` : str
                * `"manifest_hash"` : str
            Upstream metadata (counts, etc.) may also be present.
            - `is_last` : bool
            Whether this is the final batch.

        Returns
        -------
        Iterable[ManifestBatch]
            A generator over `ManifestBatch` objects. Note that this function
            **does not** produce a list; iterate to realize work.

        Notes
        -----
        - **Normalization**: `data_sources` and `label_sources` are coerced to
        serializable dicts so parameters are stable across runs.
        - **Coordinate systems**: `box_cfg.coord_in` → `box_cfg.coord_out` controls
        bounding-box representation (e.g., `xyxy_abs` → `xyxy_rel`) and clamping.
        - **Un/Labelled images**: `join_cfg.keep_unlabeled_images` determines whether
        images without labels are kept in the manifest.
        - **Tracking behavior**:
        When `track=True`, a tracking stream is opened with function name
        `"blase.Prepare.build_manifest"`. A manifest artifact is registered once
        and its identifier is injected into every emitted batch under
        `"manifest_root_hash"` and `"manifest_hash"`. This identity is later used
        by replay handlers (e.g., `split`, `to_tfrecord`) to reconstruct the same
        upstream manifest pipeline.
        - **Memory**: Emission is streaming; only `scale_cfg.batch_rows` rows are
        held in memory per batch.

        See Also
        --------
        Prepare.split : Create train/val/test splits from this manifest.
        Prepare.to_tfrecord : Serialize a manifest + splits to TFRecords.
        Prepare.write_label_sidecars : Emit per-split label files.
        Prepare.preview_tfrecord : Inspect TFRecord contents.
        Prepare.class_map_io : Manage class→id mappings.

        Examples
        --------
        Basic usage with a Parquet images table and COCO labels::

            prep = Prepare()
            manifest = prep.build_manifest(
                data_sources=[prepare.images_parquet("data/images_*.parquet")],
                label_sources=[prepare.coco("data/labels.json")],
                image_cfg=ImageConfig(id_col="image_id", sha256_col="sha256"),
                join_cfg=JoinConfig(image_id_resolver="provided", keep_unlabeled_images=True),
                box_cfg=BoxConfig(coord_in="xyxy_abs", coord_out="xyxy_rel", clamp_boxes=True),
                scale_cfg=ScaleConfig(join_strategy="duckdb_temp", batch_rows=2048),
                track=True,
            )

            for mb in manifest:
                # consume mb.data
                pass

        """
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
        """
        Compute dataset statistics from a manifest stream.

        This function consumes a streamed manifest and returns a single
        `Batch` whose `data` contains summary statistics. The exact schema
        of `data` depends on the aggregation mode selected via `by`.

        Parameters
        ----------
        manifest : Iterable[ManifestBatch]
            A generator of manifest batches as produced by
            `Prepare.build_manifest`. Each batch's `meta` may include
            `"manifest_root_hash"` / `"manifest_hash"` when tracking is enabled.
        by : {"class", "image", "global"}, default="class"
            Aggregation mode:
            - "class"  : Class-level stats (e.g., per-class instance counts).
                        Expect a mapping such as `{"class_hist": {<name>: int, ...}}`.
            - "image"  : Image-level stats (e.g., boxes per image, label presence).
                        Expect per-image summaries (implementation-defined keys).
            - "global" : Dataset-wide totals (e.g., total images/instances),
                        returned as a flat mapping of counters.
        track : bool, default=True
            If `True`, records this step for replay. The handler captures the
            upstream manifest identities from `manifest` batch metadata and
            registers them as inputs:
            - role `"manifest_root"`  → the root manifest id
            - role `"manifest"`       → the concrete manifest id

        Returns
        -------
        Batch[Stats, ManifestMeta]
            A terminal batch with:
            - `data` : `Stats` (a dict-like structure whose fields depend on `by`).
            For example, in `"class"` mode, `data["class_hist"]` is a mapping
            `{class_name: count}`.
            - `meta` : `ManifestMeta` with at least `{"by": <mode>}`. When `track`
            is `True`, the step emits once with `last_batch=True`.
            - `is_last` : `True`.

        Notes
        -----
        - The function streams the manifest once and aggregates on the fly; it
        does not materialize the entire dataset.
        - In tracked mode, upstream manifest ids are *plumbed through* by reading
        `mb.meta["manifest_root_hash"]` / `mb.meta["manifest_hash"]` from the
        incoming batches and registering them on the step. This enables exact
        replay of the stats against the same manifest lineage.

        Examples
        --------
        Compute per-class counts and extract the class names::

            prep = Prepare()
            m = prep.build_manifest(..., track=True)
            stats_b = prep.compute_stats(manifest=m, by="class", track=True)
            class_names = sorted(stats_b.data["class_hist"].keys())
        """
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
        """
        Partition a manifest stream into train/val/test splits.

        This function consumes the manifest in a single pass and assigns each
        example to one of the requested partitions according to the selected
        splitting strategy. The result is returned as a terminal `Batch` whose
        `data` is a mapping of split names to ordered identifiers (typically
        example/image ids), and whose `meta` contains split configuration and
        size summaries. When `track=True`, upstream manifest identities are
        recorded so the split can be faithfully replayed.

        Parameters
        ----------
        manifest : Iterable[ManifestBatch]
            Stream of manifest batches produced by `Prepare.build_manifest`.
            Each batch should include ids and (depending on `method`) fields
            required by the splitter (e.g., class labels, group keys, or
            timestamps). When tracking is enabled, the upstream manifest
            hashes are read from `mb.meta["manifest_root_hash"]` and
            `mb.meta["manifest_hash"]`.
        method : {"random", "stratified", "group", "time"}, default="stratified"
            Splitting strategy:

            - **"random"**:
            Uniformly assign examples to splits using `seed`.
            - **"stratified"**:
            Preserve label distribution across splits. Uses the field
            specified by `stratify_on` (default `"class"`). Multi-label
            handling follows the implementation of `_split_strat`.
            - **"group"**:
            Keep all examples with the same `group_key` together
            (e.g., `image_id`, `sequence_id`, `patient_id`) to prevent
            leakage across splits.
            - **"time"**:
            Chronologically split by a timestamp field (implementation-defined);
            typical use is to allocate early → train, middle → val, late → test.

        train, val, test : float, default=(0.8, 0.1, 0.1)
            Fractions for each partition. **Must sum to 1.0** (within 1e-6).
            If one of the splits is undesired, pass 0.0 (e.g., `test=0.0`).
        seed : int, default=42
            PRNG seed used by non-deterministic strategies ("random", "stratified")
            to ensure reproducibility.
        group_key : str, default="image_id"
            Column/field name used by the "group" splitter to assign examples
            atomically by group.
        stratify_on : str, default="class"
            Column/field used by the "stratified" splitter to preserve label
            distribution across splits.
        holdout_query : str or None, optional
            Optional filter expression to reserve a subset (e.g., a product,
            site, or domain) entirely outside the split calculation. Semantics
            are defined by the underlying splitter; when provided, matched
            examples are excluded from the main fractioning and emitted into
            their own holdout partition(s) per implementation.
        track : bool, default=True
            If `True`, records the split step and registers upstream manifest
            identities as inputs:
            - role `"manifest_root"` → root manifest id
            - role `"manifest"`      → concrete manifest id

        Returns
        -------
        Batch[Splits, ManifestMeta]
            Terminal batch where:
            - `data` (Splits): `{"train": [ids...], "val": [...], "test": [...]}`.
            Order within each split is stable and derived from the incoming
            iteration order unless the strategy defines otherwise.
            - `meta` (ManifestMeta): includes
            `{"method", "fractions", "seed", "sizes"}` and, when tracked,
            upstream manifest ids (`"upstream_manifest_root"`, `"upstream_manifest"`)
            and a registered `"splits_id"` artifact when available.
            - `is_last` is `True`.

        Raises
        ------
        ValueError
            If `train + val + test != 1.0` (within tolerance), or if `method`
            is not one of the supported values.

        Notes
        -----
        - The function streams the manifest once and does not materialize the
        entire dataset.
        - For "group" splits, all records sharing `group_key` are assigned to
        the same partition to avoid target leakage.
        - For "stratified" splits, `stratify_on` must be present in the
        manifest records (schema depends on your `build_manifest` config).
        - For "time" splits, the timestamp interpretation and ordering are
        delegated to the underlying `_split_time` implementation.
        - When `track=True`, a lightweight "splits" artifact is registered
        (metadata-only) to aid provenance without storing large payloads.

        Examples
        --------
        Basic stratified split with no test set::

            prep = Prepare()
            m = prep.build_manifest(..., track=True)
            splits_b = prep.split(
                manifest=m,
                method="stratified",
                train=0.9, val=0.1, test=0.0,
                seed=123, stratify_on="class", track=True
            )
            len_train = len(splits_b.data["train"])

        Group-aware split to keep sequences together::

            splits_b = prep.split(
                manifest=m,
                method="group",
                group_key="sequence_id",
                train=0.8, val=0.2, test=0.0,
            )
        """
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
        """
        Write a manifest stream to TFRecord files, partitioned by `splits`.

        This function consumes a streaming manifest (as produced by
        `Prepare.build_manifest`) and emits one or more `.tfrecord` shards per
        split (train/val/test). Optionally writes a compact alignment index per
        shard for reproducible re-reading. When `track=True`, the step records
        parameters, the upstream manifest identity, the `splits` blob, and the
        set of produced files so the exact write can be restored later.

        Parameters
        ----------
        manifest : Iterable[ManifestBatch]
            Stream of `ManifestBatch` objects containing example-level features
            (e.g., `image_id`, geometry/labels, optional `sha256`) and per-batch
            metadata (`mb.meta["manifest_root_hash"]`, `mb.meta["manifest_hash"]`
            if tracking was enabled upstream).
        splits : dict-like
            Mapping of split name → ordered list/iterable of example IDs
            (e.g., `{"train": [...], "val": [...], "test": [...]}`).
            Only examples whose IDs appear in a split are written to that split.
        out_dir : pathlib.Path
            Target directory where TFRecord files (and optional index files) are
            created. The directory is created if it does not exist.
        mode : {"combined", "images_labels"}, default "combined"
            Controls how examples are serialized:
            - "combined": a single Example per record with image + label features.
            - "images_labels": emits image-only and label-only streams (writer
            implementation determines exact layout and filenames).
        shard_size_mb : int, default 128
            Approximate uncompressed shard size in MiB. Writers will rotate to a
            new shard when the current shard reaches this threshold.
        compression : {"GZIP", "NONE"}, default "GZIP"
            Compression codec used for TFRecord files.
        include_image_bytes : bool, default True
            If True, raw image bytes are embedded in the record (when available).
            If False, only metadata/paths are written (exact fields depend on the
            writer and the `mode`).
        read_bytes_from : {"parquet", "filesystem"}, default "parquet"
            Source for loading image bytes when `include_image_bytes=True`:
            - "parquet": read from a column in the input parquet rows.
            - "filesystem": read from `path`/`filepath` on disk.
        parquet_id_col : str, default "image_id"
            Column name in the parquet view that contains the example identifier.
            Used to join with `splits` and to populate `example_id_feature`.
        parquet_bytes_col : str, default "img_bytes"
            Column name in the parquet view that contains the raw image bytes
            (when `read_bytes_from="parquet"`).
        deterministic_order : bool, default True
            If True, examples within a shard are written in a stable order
            derived from `order_key`. If False, the writer may use a faster,
            non-stable ordering.
        order_key : {"sha256", "image_id"}, default "sha256"
            Key used to order examples deterministically within each split.
            Requires the corresponding field to be present in the manifest.
        write_workers : int or None, optional
            Optional parallelism hint for the writer implementation. `None`
            lets the writer choose a default.
        write_alignment_index : bool, default True
            If True, writes an alignment index (e.g., `.index` files) alongside
            TFRecord shards to support reproducible mapping between example IDs and
            record offsets.
        example_id_feature : str, default "example/id"
            Feature name under which the example identifier is stored in each
            record (string feature).
        track : bool, default True
            If True, record a tracked step:
            - Params are captured (with paths stringified) for replay.
            - Upstream manifest identities (root + concrete) are recorded as
            inputs when present in `ManifestBatch.meta`.
            - The `splits` mapping is serialized and registered as a blob input.
            - Each produced file (TFRecord and index) is registered as an output.

        Returns
        -------
        SinkResult
            An object with:
            - `artifacts`: list of small records `{path, kind}` for produced files.
            `kind` is `"tfrecord"` or `"tfrecord.index"` when applicable.
            - `meta`: writer-defined metadata (e.g., shard counts, sizes).
            - `is_last`: always `True` for sinks.

        Raises
        ------
        KeyError
            If required columns (e.g., `parquet_bytes_col` when reading bytes from
            parquet) are missing from the manifest/parquet schema.
        ValueError
            If writer- or mode-specific validation fails (e.g., unknown `mode`).
        OSError / IOError
            If output files cannot be created or written.

        Notes
        -----
        - The function is streaming and does not need to materialize the full
        manifest. However, deterministic ordering may introduce buffering as
        required by the writer.
        - `splits` governs which examples are written to which shard groups.
        Examples not present in any split are skipped.
        - When `track=True`, a serialized `splits.json` blob is registered as
        an input so the step can be restored without requiring a separate
        `splits.json` file on disk.

        Examples
        --------
        Basic combined TFRecords with embedded bytes from parquet::

            prep = Prepare()
            m = prep.build_manifest(..., track=True)
            splits_b = prep.split(manifest=m, method="stratified", seed=123, track=True)

            res = prep.to_tfrecord(
                manifest=prep.build_manifest(..., track=True),  # fresh stream
                splits=splits_b.data,
                out_dir=Path("out/tfrecord"),
                include_image_bytes=True,
                read_bytes_from="parquet",
                parquet_id_col="image_id",
                parquet_bytes_col="img_bytes",
                order_key="image_id",
                track=True,
            )

        Writing label/image streams separately and disabling index files::

            res = prep.to_tfrecord(
                manifest=m,
                splits=splits_b.data,
                out_dir=Path("out/tfrecord"),
                mode="images_labels",
                write_alignment_index=False,
                compression="NONE",
            )
        """
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
        track: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Preview a TFRecord shard and optionally materialize thumbnail images.

        Reads up to the first ``n`` examples from a TFRecord file, parses common
        fields, and returns a small, human-inspectable summary for each example.
        When ``decode_images=True`` and an ``out_dir`` is provided, decoded image
        thumbnails (PNG) are written alongside the returned metadata to aid quick
        sanity checks.

        Parameters
        ----------
        path : pathlib.Path
            Path to a TFRecord file (e.g., ``train-00000.tfrecord``).
        n : int, default=8
            Maximum number of examples to preview from the shard. If the file
            contains fewer than ``n`` records, all available records are read.
        compression : {"GZIP", "NONE"}, default="GZIP"
            Compression codec used by the TFRecord file.
        decode_images : bool, default=False
            If ``True``, attempt to decode image bytes found in each example and
            write a thumbnail (PNG) to ``out_dir``. If ``False``, only metadata is
            returned; no files are written.
        out_dir : pathlib.Path or None, optional
            Directory where preview thumbnails are written when
            ``decode_images=True``. The directory is created if it does not exist.
            Ignored when ``decode_images=False``.
        max_side : int, default=1024
            Max dimension (pixels) for the longest side of each written thumbnail.
            The shorter side is scaled to preserve aspect ratio.
        track : bool, default=True
            If ``True``, record a tracked step with:
            - the source TFRecord hash (as an input),
            - the set of written thumbnail files (as outputs),
            - a small JSON summary blob (count, codec, decode flag, paths, etc.).
            This enables step and data restoration via the CLI.

        Returns
        -------
        list of dict
            A list of per-example dictionaries containing a concise view of
            parsed fields. The exact keys are reader-dependent, but commonly
            include identifiers (e.g., ``"example/id"``), dimensions, and any
            readily interpretable label fields. When thumbnails are written, each
            dict may include a relative filename of the generated preview.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If ``compression`` is not supported or the TFRecord cannot be parsed.
        OSError
            If thumbnails cannot be written to ``out_dir`` (e.g., permission or
            filesystem errors).

        Notes
        -----
        - Only a *preview* of the shard is read; the function does not scan the
        entire file unless ``n`` exceeds the number of available records.
        - Thumbnails are written with the ``*.preview.png`` suffix when
        ``decode_images=True`` and ``out_dir`` is provided.
        - This method does **not** write image bytes to CAS; it registers outputs
        by file path when tracking is enabled so they can be restored later.

        Examples
        --------
        Basic preview (no thumbnails)::

            prep = Prepare()
            rows = prep.preview_tfrecord(
                path=Path("out/tfrecord/train-00000.tfrecord"),
                n=5,
                compression="GZIP",
                decode_images=False,
                track=True,
            )

        Preview with thumbnails written to a directory::

            thumbs = Path("out/tfrecord_preview")
            rows = prep.preview_tfrecord(
                path=Path("out/tfrecord/train-00000.tfrecord"),
                n=8,
                compression="GZIP",
                decode_images=True,
                out_dir=thumbs,
                max_side=512,
                track=True,
            )
        """
        cfg = {
            "path": str(Path(path)),
            "n": int(n),
            "compression": compression,
            "decode_images": bool(decode_images),
            "out_dir": (str(out_dir) if out_dir else None),
            "max_side": int(max_side),
        }

        tracker = Track.get(track)
        if tracker is None:
            return _tfr_head(
                path=cfg["path"],
                n=cfg["n"],
                compression=cfg["compression"],
                decode_images=cfg["decode_images"],
                out_dir=(Path(cfg["out_dir"]) if cfg["out_dir"] else None),
                max_side=cfg["max_side"],
            )

        stream = tracker.stream(
            "blase.Prepare.preview_tfrecord", cfg, code_fn=self.preview_tfrecord
        )
        try:
            tf_path = Path(cfg["path"])
            src_hash = Hash().hash_file(tf_path)
            try:
                stream.step.add_input(src_hash, role="source", arg_name="path")
            except Exception:
                pass

            out = _tfr_head(
                path=tf_path,
                n=cfg["n"],
                compression=cfg["compression"],
                decode_images=cfg["decode_images"],
                out_dir=(Path(cfg["out_dir"]) if cfg["out_dir"] else None),
                max_side=cfg["max_side"],
            )

            written_files = []
            if cfg["decode_images"] and cfg["out_dir"]:
                for p in sorted(Path(cfg["out_dir"]).glob("*.preview.png")):
                    try:
                        dh = stream.step.register_data(
                            kind="image",
                            version="1",
                            path_or_bytes=str(p),
                            metadata={"path": str(p), "source": src_hash},
                        )
                        stream.step.add_output(dh, name=p.name)
                        written_files.append(str(p))
                    except Exception:
                        pass

            summary = {
                "n": cfg["n"],
                "compression": cfg["compression"],
                "decode_images": cfg["decode_images"],
                "out_dir": cfg["out_dir"],
                "max_side": cfg["max_side"],
                "written": written_files,
            }
            try:
                b = json.dumps(summary, ensure_ascii=False).encode("utf-8")
                _ = stream.step.register_data(
                    kind="preview.summary",
                    version="1",
                    path_or_bytes=b,
                    metadata=summary,
                )
            except Exception:
                pass

            stream.emit(
                last_batch=True,
                meta={"source": src_hash, "written": written_files},
            )
            stream.close_ok()
            return out
        except Exception as e:
            stream.close_error(type(e), e, e.__traceback__)
            raise

    def write_label_sidecars(
        self,
        manifest: Iterable[ManifestBatch],
        splits: Splits,
        *,
        out_dir: Path,
        format: Literal["jsonl", "parquet"] = "jsonl",
        deterministic_order: bool = True,
        order_key: Literal["sha256", "image_id"] = "sha256",
        track: bool = True,
    ) -> SinkResult:
        """
        Materialize per-split label sidecar files (JSONL or Parquet).

        Given a manifest stream and a ``splits`` mapping (e.g., ``{"train": [...], "val": [...], ...}``),
        this function writes label-only “sidecar” files for each split into ``out_dir``.
        The sidecars contain the labels (and any required identifiers) in a flat, analysis-friendly
        format that mirrors the examples present in each split.

        Parameters
        ----------
        manifest : Iterable[ManifestBatch]
            A manifest stream produced by :meth:`build_manifest`. Each batch must carry
            manifest identity in ``mb.meta["manifest_root_hash"]`` and ``mb.meta["manifest_hash"]``
            when tracking is enabled (the builder sets these).
        splits : dict
            A mapping of split name to a list/sequence of example identifiers. The expected
            identifier depends on your pipeline (commonly ``"image_id"`` or a sha256).
            This should be the same object previously returned by :meth:`split`.
        out_dir : pathlib.Path
            Target directory where sidecar files are written. Created if it does not exist.
            Files are named ``{split}.labels.{jsonl|parquet}``.
        format : {"jsonl", "parquet"}, default="jsonl"
            Output file format for sidecars. Use ``"jsonl"`` for line-delimited JSON or
            ``"parquet"`` for a columnar, compressed table.
        deterministic_order : bool, default=True
            If ``True``, examples are emitted in a stable order so materialized files are
            byte-for-byte reproducible for a fixed manifest and splits.
        order_key : {"sha256", "image_id"}, default="sha256"
            Primary key used to order examples when ``deterministic_order=True``.
        track : bool, default=True
            If ``True``, records a tracked step:
            - Registers the splits payload as a first-class input (``kind="splits.json"``).
            - Adds upstream manifest identities (``role="manifest_root"`` and ``"manifest"``) when present.
            - Registers each produced sidecar file as an output (``kind="labels.sidecar.jsonl"`` or
            ``"labels.sidecar.parquet"``) with its path metadata.
            This enables step replay and individual data restoration via the CLI.

        Returns
        -------
        SinkResult
            An object with:
            - ``artifacts``: list of produced files (with ``.path`` and optional ``.kind``),
            - ``is_last``: always ``True`` for this sink,
            - ``meta``: implementation-specific metadata (e.g., counts per split).

        Raises
        ------
        ValueError
            If ``format`` is not one of the supported values or output writing fails due to
            invalid configuration (e.g., missing columns required by the writer).
        OSError
            If files cannot be created in ``out_dir`` (permissions, disk full, etc.).

        Notes
        -----
        - **Determinism**: With ``deterministic_order=True``, the combination of ``order_key`` and
        a stable manifest yields reproducible file contents.
        - **Tracking & Replay**: When tracking is enabled, the function also persists the splits JSON
        bytes into the run’s CAS so that ``blase restore run --step … --mode replay`` can reconstruct
        the same sidecars without the original Python process. Individual files can be restored by
        their recorded data hash via ``blase restore run --data <hash>``.
        - **Schema**:
        - JSONL: one JSON object per line; minimally includes the join key (e.g., ``image_id``)
            and the label payload for each example in the split.
        - Parquet: equivalent columns in columnar form for efficient downstream reads.

        Examples
        --------
        Write JSONL sidecars for train/val/test::

            prep = Prepare()
            manifest = prep.build_manifest(...)
            splits = prep.split(manifest, method="stratified", train=0.8, val=0.1, test=0.1).data

            res = prep.write_label_sidecars(
                manifest=manifest,
                splits=splits,
                out_dir=Path("out/sidecars"),
                format="jsonl",
                order_key="image_id",
                track=True,
            )

        Write Parquet sidecars in deterministic order::

            res = prep.write_label_sidecars(
                manifest=manifest,
                splits=splits,
                out_dir=Path("out/sidecars_parquet"),
                format="parquet",
                deterministic_order=True,
                order_key="sha256",
            )

        See Also
        --------
        Prepare.split : Produce the ``splits`` mapping consumed here.
        Prepare.to_tfrecord : Emit TFRecord shards for training; sidecars complement these for analysis.
        """
        cfg: Dict[str, Any] = {
            "out_dir": Path(out_dir),
            "deterministic_order": bool(deterministic_order),
            "order_key": order_key,
            "format": format,
        }

        tracker = Track.get(track)
        if tracker is None:
            try:
                writer = _sidecar_jsonl if format == "jsonl" else _sidecar_parquet
                return writer.write(manifest, splits, cfg)
            except Exception:
                raise ValueError(f"unsupported sidecar format: {format!r}")

        params = {**cfg, "out_dir": str(Path(out_dir))}
        stream = tracker.stream(
            "blase.Prepare.write_label_sidecars",
            params,
            code_fn=self.write_label_sidecars,
        )

        upstream = {"manifest_root_hash": None, "manifest_hash": None}

        def _iter_manifest():
            for mb in manifest:
                rh = (mb.meta or {}).get("manifest_root_hash")
                mh = (mb.meta or {}).get("manifest_hash")
                upstream["manifest_root_hash"] = upstream["manifest_root_hash"] or rh
                upstream["manifest_hash"] = upstream["manifest_hash"] or mh
                yield mb

        try:
            splits_json = json.dumps(splits, sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        except Exception:
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
            splits_hash = None

        writer = _sidecar_jsonl if format == "jsonl" else _sidecar_parquet
        try:
            res = writer.write(_iter_manifest(), splits, cfg)

            for rid, role in [
                (upstream["manifest_root_hash"], "manifest_root"),
                (upstream["manifest_hash"], "manifest"),
            ]:
                if rid:
                    try:
                        stream.step.add_input(rid, role=role, arg_name=None)
                    except Exception:
                        pass

            artifacts = res.artifacts or []
            for i, a in enumerate(artifacts, 1):
                kind = (
                    "labels.sidecar.parquet"
                    if str(a.path).endswith(".parquet")
                    else "labels.sidecar.jsonl"
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

            meta = dict(res.meta or {})
            new_meta = stream.emit(last_batch=True, meta=meta) or meta
            stream.close_ok()
            return SinkResult(artifacts=artifacts, is_last=res.is_last, meta=new_meta)

        except Exception as e:
            stream.close_error(type(e), e, e.__traceback__)
            raise

    def class_map_io(
        self,
        *,
        load: Optional[Path] = None,
        save: Optional[Path] = None,
        map: Optional[Dict[str, int]] = None,
        normalize: bool = True,
        track: bool = True,
    ) -> Batch[Dict[str, int], ManifestMeta]:
        """
        Load and/or persist a canonical class to id mapping with optional tracking.

        This utility consolidates a class map from one of three sources—an explicit
        ``map`` argument, a JSON file at ``load``, or an empty/default mapping—then
        normalizes class names to a canonical form and optionally writes the result
        to ``save``. When ``track=True``, the operation is recorded for reproducible
        step and data restoration.

        Parameters
        ----------
        load : pathlib.Path, optional
            Path to a JSON file containing a class→id mapping. Used only if
            ``map`` is not provided. The file must be UTF-8 encoded and valid JSON.
        save : pathlib.Path, optional
            Destination path to write the normalized class map as pretty-printed
            JSON. Parent directories are created if needed.
        map : dict[str, int], optional
            In-memory class→id mapping. Takes precedence over ``load``. The mapping
            will be normalized and validated.
        normalize : bool, default=True
            If ``True``, class names are normalized (e.g., case folding, whitespace
            trimming, stable canonicalization) before validation and persistence.
        track : bool, default=True
            If ``True``, records a tracked step:
            - Persists the inline ``map`` (when provided) as a data blob
            (``kind="class.map.json"``) and registers it as an input.
            - If ``load`` is used, registers the file hash as an input.
            - If ``save`` is provided, registers the written file as an output
            (``kind="class.map.file"``).
            This enables replay via ``blase restore run --step … --mode replay`` and
            individual data restoration via ``--data <hash>``.

        Returns
        -------
        Batch[dict[str, int], ManifestMeta]
            A single terminal ``Batch`` whose ``data`` is the normalized class map and
            whose ``meta`` includes:
            - ``size`` : int — number of classes
            - ``normalized`` : bool — whether normalization was applied
            - ``hash`` : str — stable hash of the normalized map JSON
            - ``loaded_from`` : str | None — source path when ``load`` was used
            - ``saved_to`` : str | None — destination path when ``save`` was used

        Raises
        ------
        json.JSONDecodeError
            If the file at ``load`` is not valid JSON.
        OSError
            If the file at ``load`` cannot be read or the file at ``save`` cannot
            be written (permissions, missing directories beyond creatable parents,
            disk full, etc.).
        ValueError
            If the provided mapping is invalid (e.g., non-string keys, non-int
            ids) after normalization.

        Notes
        -----
        - **Precedence**: ``map`` > ``load`` > empty mapping.
        - **Determinism**: The output JSON (when saving) is produced with sorted
        keys and a stable canonicalization, enabling content-hash reproducibility.
        - **Replay**: With tracking enabled, inline maps are also written into CAS
        so restore can replay without the original in-process object.

        Examples
        --------
        Normalize and save a class map::

            prep = Prepare()
            cm = {"Dog": 0, " cat  ": 1}
            out = prep.class_map_io(save=Path("data/working/classmap/class_map.json"),
                                    map=cm, normalize=True)
            print(out.data)  # e.g., {'cat': 1, 'dog': 0}

        Load from a file, do not write, return normalized map (no tracking)::

            out = prep.class_map_io(load=Path("data/working/classmap/class_map.json"),
                                    normalize=True, track=False)

        Track the operation for later restore::

            out = prep.class_map_io(save=Path("data/working/classmap/class_map.json"),
                                    map=cm, track=True)

        Integrate with ``build_manifest`` to lock class IDs::

            # Prepare helpers (example)
            from blase.prepare import ImageConfig, JoinConfig, BoxConfig, ClassConfig, ScaleConfig
            from blase.prepare import images_parquet, LabelSource

            prep = Prepare()

            def manifest_with_fixed_ids():
                # 1) Load the canonical class map (tracked or untracked)
                cm = prep.class_map_io(
                    load=Path("data/working/classmap/class_map.json"),
                    normalize=True,
                ).data

                # 2) Build the manifest while *locking* class ids via class_cfg.class_map
                return prep.build_manifest(
                    data_sources=[images_parquet("data/working/images_*.parquet")],
                    label_sources=[
                        LabelSource(
                            kind="detection",
                            uri="data/working/labels.ndjson",
                            fmt="labelbox",
                            options={"id_from": "external_id", "mode": "detection"},
                        )
                    ],
                    image_cfg=ImageConfig(
                        id_col="filename",
                        path_col="path",
                        bytes_col="img_bytes",
                        height_col="height",
                        width_col="width",
                        sha256_col=None,
                    ),
                    join_cfg=JoinConfig(
                        image_id_resolver="provided",
                        drop_orphans=True,
                        keep_unlabeled_images=True,
                    ),
                    box_cfg=BoxConfig(
                        coord_in="xywh_abs",
                        coord_out="xyxy_rel",
                        clamp_boxes=True,
                        drop_oob_boxes=False,
                    ),
                    class_cfg=ClassConfig(class_map=cm),  # ← lock IDs deterministically
                    scale_cfg=ScaleConfig(join_strategy="duckdb_temp", batch_rows=2048, seed=42),
                    track=True,
                )

        See Also
        --------
        Prepare.build_manifest : Build a manifest stream; accepts a fixed ``class_map``.
        Prepare.split : Produces splits that often rely on canonical class ids.
        """
        tracker = Track.get(track)

        if tracker is None:
            if map is not None:
                cm = dict(map)
            elif load is not None:
                txt = Path(load).read_text(encoding="utf-8")
                cm = json.loads(txt)
            else:
                cm = {}

            cm_norm = _classmap.canonicalize_class_map(cm, normalize_names=normalize)
            if save is not None:
                sp = Path(save)
                sp.parent.mkdir(parents=True, exist_ok=True)
                sp.write_text(
                    json.dumps(cm_norm, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            j = json.dumps(cm_norm, sort_keys=True, ensure_ascii=False).encode("utf-8")
            meta: ManifestMeta = {
                "size": len(cm_norm),
                "normalized": bool(normalize),
                "hash": Hash().hash_object(j),
                "loaded_from": str(load) if load else None,
                "saved_to": str(save) if save else None,
            }
            return Batch(data=cm_norm, is_last=True, meta=meta)

        params = {
            "load": (str(load) if load else None),
            "save": (str(save) if save else None),
            "normalize": bool(normalize),
            # do NOT inline the map here; persist it as a blob instead
            "has_inline_map": map is not None,
        }
        stream = tracker.stream(
            "blase.Prepare.class_map_io", params, code_fn=self.class_map_io
        )
        try:
            # 1) assemble source & record inputs
            cm = None
            # inline map → persist as CAS blob (input role='map')
            if map is not None:
                map_bytes = json.dumps(map, ensure_ascii=False).encode("utf-8")
                try:
                    map_hash = stream.step.register_data(
                        kind="class.map.json",
                        version="1",
                        path_or_bytes=map_bytes,
                        metadata={"size": len(map)},
                    )
                    # ensure the blob exists on disk for replay
                    try:
                        cas.write_bytes(
                            run_path=stream.step.run_path,
                            kind="data",
                            data_hash=map_hash,
                            data=map_bytes,
                        )
                    except Exception:
                        pass
                    stream.step.add_input(map_hash, role="map", arg_name="map")
                except Exception:
                    map_hash = None
                cm = dict(map)

            # load=... → record the source file hash (input role='source')
            if cm is None and load is not None:
                p = Path(load)
                try:
                    src_hash = Hash().hash_file(p)
                    stream.step.add_input(src_hash, role="source", arg_name="load")
                except Exception:
                    pass
                txt = p.read_text(encoding="utf-8")
                cm = json.loads(txt)

            if cm is None:
                cm = {}

            # 2) normalize
            cm_norm = _classmap.canonicalize_class_map(cm, normalize_names=normalize)

            # 3) persist if requested; also record file output in CAS index
            out_path = None
            if save is not None:
                out_path = Path(save)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(cm_norm, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                try:
                    out_hash = stream.step.register_data(
                        kind="class.map.file",
                        version="1",
                        path_or_bytes=str(out_path),
                        metadata={"path": str(out_path)},
                    )
                    stream.step.add_output(out_hash, name="class_map_json")
                except Exception:
                    pass

            # 4) meta + emit
            j = json.dumps(cm_norm, sort_keys=True, ensure_ascii=False).encode("utf-8")
            meta = {
                "size": len(cm_norm),
                "normalized": bool(normalize),
                "hash": Hash().hash_object(j),
                "loaded_from": str(load) if load else None,
                "saved_to": str(save) if save else None,
            }
            stream.emit(last_batch=True, meta=meta)
            stream.close_ok()
            return Batch(data=cm_norm, is_last=True, meta=meta)

        except Exception as e:
            stream.close_error(type(e), e, e.__traceback__)
            raise

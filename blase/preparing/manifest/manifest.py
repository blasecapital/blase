from typing import Any, Dict, Iterable, Sequence, Callable, List
from ..interfaces import ManifestBatch
from ..registry import PrepareRegistry


def _mk_label_readers(
    reg: PrepareRegistry, srcs: Sequence[Dict[str, Any]]
) -> List[Callable[[], Iterable[Dict[str, Any]]]]:
    fns = []
    for s in srcs:
        fmt = s["fmt"]
        reader = reg.label_readers.get(fmt)
        if reader is None:
            raise ValueError(f"no label reader for fmt={fmt!r}")
        payload = {"uri": s["uri"], "fmt": fmt, "options": dict(s.get("options", {}))}

        def _mk(p=payload, r=reader):
            def _reader():
                return r.read(p)

            return _reader

        fns.append(_mk())
    return fns


def build_manifest_stream(
    *,
    reg: PrepareRegistry,
    data_sources: Sequence[Dict[str, Any]],
    label_sources: Sequence[Dict[str, Any]],
    image_cfg: Dict[str, Any],
    join_cfg: Dict[str, Any],
    box_cfg: Dict[str, Any],
    class_cfg: Dict[str, Any],
    scale_cfg: Dict[str, Any],
) -> Iterable[ManifestBatch]:
    kv_index, _bloom, rg_index = reg.image_indexer.build_index(
        [
            {"uri": d["uri"], "fmt": d["fmt"], "options": dict(d.get("options", {}))}
            for d in data_sources
        ],
        {
            "rows_per_chunk": scale_cfg.get("rows_per_chunk"),
            "index_backend": scale_cfg.get("index_backend"),
            "id_col": image_cfg.get("id_col"),
            "path_col": image_cfg.get("path_col"),
            "height_col": image_cfg.get("height_col"),
            "width_col": image_cfg.get("width_col"),
            "sha256_col": image_cfg.get("sha256_col"),
        },
    )
    label_fns = _mk_label_readers(reg, label_sources)

    # class map first pass
    cm, class_meta = reg.classmap.build_or_validate(
        [fn() for fn in label_fns],
        {
            "class_map": class_cfg.get("class_map"),
            "normalize_names": class_cfg.get("normalize_names"),
        },
    )

    # optional probe for fast fail
    kv_ids = set(kv_index.keys())
    probe_ids = set()
    for it in [fn() for fn in label_fns]:
        for i, row in enumerate(it):
            iid = row.get("image_id")
            if iid:
                probe_ids.add(iid)
            if i >= 999:
                break
    if probe_ids and not (kv_ids & probe_ids):
        raise RuntimeError("no label image_ids match Parquet ids; check id_from/id_col")

    # second pass: align
    iters = [fn() for fn in label_fns]
    yield from reg.aligner.align_stream(
        label_iters=iters,
        kv_index=kv_index,
        rg_index=rg_index,
        join_cfg=join_cfg,
        box_cfg=box_cfg,
        class_cfg={
            "class_map": cm,
            "normalize_names": class_cfg.get("normalize_names"),
        },
        scale_cfg=scale_cfg,
    )

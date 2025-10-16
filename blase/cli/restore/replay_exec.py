from pathlib import Path
from typing import Optional

from blase.restoring import store, bindings, cas, code
from blase.utils.config import RESTORE_DEFAULT_DIR
from blase.utils.hashing import Hash


# ==========
# Helpers
# ==========
def _ord_suffix_int(name: str, condition: Optional[str], default: int = 0) -> int:
    try:
        if condition == "parquet_shard":
            return int(name.split("_")[-1]) if name.startswith("parquet_shard_") else 0
        return int(str(name).rsplit("_", 1)[-1])
    except Exception:
        return default


def _pick_anchor(ins, roles=("source", "manifest", "dataset", "manifest_root")):
    for r in roles:
        h = next((i["data_hash"] for i in ins if i["role"] == r), None)
        if h:
            return r, h
    return None, None


# ==========
# read_csv
# ==========
def _exec_extract_csv(run_path, sh, st):
    ins = store.load_step_inputs(run_path, sh)
    src_hash = next((i["data_hash"] for i in ins if i["role"] == "source"), None)
    realized = {}
    if src_hash:
        realized["__expected_source_hash__"] = src_hash
    return bindings.run_read_csv_restore(
        run_path=run_path,
        params=st["params"],
        realized=realized,
        transform_fn=None,
    )


# ==========
# read_images
# ==========
def _exec_extract_images(run_path, sh, st):
    _ = store.load_step_inputs(run_path, sh)  # code/env only
    outs = store.load_step_outputs(run_path, sh)  # manifest + batch_desc_*

    # recorded artifacts come from outputs
    manifest_hash = next(
        (o["data_hash"] for o in outs if o["name"] == "manifest"), None
    )

    batch_descs = [
        o["data_hash"]
        for o in sorted(
            outs,
            key=lambda o: _ord_suffix_int(o.get("name", ""), condition="batch_desc"),
        )
        if o.get("name", "").startswith("batch_desc_")
    ]

    realized = {}
    if "directory" in st["params"]:
        realized["source"] = st["params"]["directory"]
    if manifest_hash:
        realized["manifest"] = manifest_hash
    if batch_descs:
        realized["batch_descs"] = batch_descs

    return bindings.run_read_images_restore(
        run_path=run_path,
        params=st["params"],
        realized=realized,
        transform_fn=None,
    )


# ==========
# read_parquet
# ==========
def _exec_extract_parquet(run_path, sh, st):
    _ = store.load_step_inputs(run_path, sh)  # code/env only (unused here)
    outs = store.load_step_outputs(run_path, sh)  # manifest + batch_desc_*

    manifest_hash = next(
        (o["data_hash"] for o in outs if o["name"] == "manifest"), None
    )

    batch_descs = [
        o["data_hash"]
        for o in sorted(
            outs,
            key=lambda o: _ord_suffix_int(o.get("name", ""), condition="batch_desc"),
        )
        if o.get("name", "").startswith("batch_desc_")
    ]

    realized = {}
    # Prefer recorded sources list; fall back to single source if present.
    if "sources" in st["params"]:
        realized["sources"] = st["params"]["sources"]
    elif "source" in st["params"]:
        realized["source"] = st["params"]["source"]

    if manifest_hash:
        realized["manifest"] = manifest_hash
    if batch_descs:
        realized["batch_descs"] = batch_descs

    return bindings.run_read_parquet_restore(
        run_path=run_path,
        params=st["params"],
        realized=realized,
        transform_fn=None,
    )


# ==========
# apply_function
# ==========
def _exec_transform_apply_function(
    run_path, upstream, sh, st, seen_steps, created_paths
):
    ins = store.load_step_inputs(run_path, sh)
    fn = code.load_callable_from_blob(
        cas.path_for(run_path, "code", store.pick_code_hash(ins))
    )

    # If an upstream generator already exists, consume it.
    if upstream is not None:
        return bindings.run_apply_function_restore(
            run_path=run_path,
            params=st["params"],
            realized={"source_gen": upstream},
            transform_fn=fn,
        )

    role, anchor_hash = _pick_anchor(ins=ins)
    if not anchor_hash:
        raise SystemExit(
            "replay: Transform.apply_function missing anchor (source|manifest|dataset)"
        )

    if role == "source":
        from .replay_cas import _resolve_source_no_copy

        src_path, _ = _resolve_source_no_copy(
            run_path,
            anchor_hash,
            seen_steps=seen_steps,
            created_paths=created_paths,
        )
        return bindings.run_apply_function_restore(
            run_path=run_path,
            params=st["params"],
            realized={"source": src_path},
            transform_fn=fn,
        )

    # treat anchor as a manifest; prefer parquet if present, else images
    man_hash = anchor_hash
    ts_before = None
    if isinstance(st, dict):
        ts_before = st.get("ts_start")
    else:
        ts_before = getattr(st, "ts_start", None)

    # Try parquet first
    rp_params = getattr(
        store, "read_parquet_params_for_manifest", lambda *a, **k: None
    )(run_path, man_hash, ts_before=ts_before)
    if rp_params:
        ins_tr = store.load_step_inputs(run_path, sh)
        batch_descs = [
            i["data_hash"]
            for i in ins_tr
            if i["role"]
            in (
                "batch_desc",
                "table.batch.meta",
                "batchmeta",
                "batch",
                "table.batch",
            )
        ]
        ins2 = store.load_step_inputs(run_path, sh)
        manifest_hash = next(
            (i["data_hash"] for i in ins2 if i.get("role") == "manifest"), None
        )
        if not manifest_hash:
            raise SystemExit("restore: cannot resolve manifest for upstream stream")
        from .anchors import _restore_gen_for_manifest

        gen = _restore_gen_for_manifest(run_path, manifest_hash, sh, store, bindings)
        return bindings.run_apply_function_restore(
            run_path=run_path,
            params=st["params"],
            realized={"source_gen": gen},
            transform_fn=fn,
        )

    # Fallback to images
    ri_params = store.read_images_params_for_manifest(
        run_path, man_hash, ts_before=ts_before
    )
    if not ri_params:
        raise SystemExit(
            "replay: cannot find read_parquet or read_images params for manifest"
        )

    ins_tr = store.load_step_inputs(run_path, sh)
    batch_descs = [
        i["data_hash"]
        for i in ins_tr
        if i["role"] in ("batch_desc", "table.batch.meta", "batch", "table.batch")
    ]
    gen = bindings.run_read_images_restore(
        run_path=run_path,
        params=ri_params,
        realized={
            "source": ri_params.get("directory"),
            "manifest": man_hash,
            "batch": batch_descs,
        },
    )
    return bindings.run_apply_function_restore(
        run_path=run_path,
        params=st["params"],
        realized={"source_gen": gen},
        transform_fn=fn,
    )


# ==========
# save_to_csv
# ==========
def _exec_load_save_to_csv(
    run_path,
    sh,
    produced_hashes,
    seen_steps,
    created_paths,
    to_path,
    st,
    backend_override,
    ephemeral_only,
    produced,
    upstream,
):
    ins = store.load_step_inputs(run_path, sh)
    outs = store.load_step_outputs(run_path, sh)
    expected_out_hash = outs[0]["data_hash"] if outs else None

    if expected_out_hash and expected_out_hash in produced_hashes:
        return

    # Pre-seed (append semantics)
    preseed_path = None
    from .replay_cas import _pick_viable_seed, _resolve_seed_no_copy_or_ephemeral

    seed_hash = _pick_viable_seed(run_path, ins)
    if seed_hash:
        preseed_path, created_seed = _resolve_seed_no_copy_or_ephemeral(
            run_path,
            seed_hash,
            seen_steps=seen_steps,
            created_paths=created_paths,
        )
        if created_seed and created_paths is not None:
            created_paths.append(preseed_path)

    # Deterministic final target
    if to_path:
        target_override_this_sink = to_path
        conflict_policy = "overwrite"
    elif expected_out_hash:
        target_override_this_sink = str(
            (RESTORE_DEFAULT_DIR / f"{expected_out_hash}.csv").resolve()
        )
        conflict_policy = "overwrite"
    else:
        default_name = Path(st["params"]["target"]).name
        target_override_this_sink = str((RESTORE_DEFAULT_DIR / default_name).resolve())
        conflict_policy = "overwrite"

    upstream_gen = upstream
    if upstream_gen is None:
        from .replay import _upstream_gen_for_sink

        upstream_gen = _upstream_gen_for_sink(run_path, sh)

    res = bindings.run_save_to_csv_replay(
        run_path=run_path,
        params=st["params"],
        realized={},
        upstream_gen=upstream_gen,
        target_override=target_override_this_sink,
        backend_override=backend_override,
        preseed_path=preseed_path,
        expected_out_hash=expected_out_hash,
        on_conflict=conflict_policy,
        record_materialization=(not ephemeral_only),
    )
    print(f"Replayed: {res}")

    if expected_out_hash:
        outp = Path(res.artifacts[0].path)
        produced[expected_out_hash] = outp
        created_paths.append(outp)
        produced_hashes.add(expected_out_hash)

    return None


# ==========
# save_images_to_parquet
# ==========
def _exec_load_save_images_to_parquet(
    run_path,
    sh,
    to_path,
    upstream,
    fqn,
    ephemeral_only,
    produced,
    created_paths,
    produced_hashes,
):
    _ = store.load_step_inputs(run_path, sh)
    outs = store.load_step_outputs(run_path, sh)
    st = store.load_step(run_path, sh)

    outs.sort(
        key=lambda o: _ord_suffix_int(o.get("name", ""), condition="parquet_shard")
    )

    expected_out_hashes = [
        o["data_hash"] for o in outs if o["name"].startswith("parquet_shard")
    ]

    if to_path:
        target_dir_override = to_path
        conflict_policy = "overwrite"
    else:
        # default: deterministic directory under RESTORE_DEFAULT_DIR
        # using the first expected hash as a hint to make the path unique.
        RESTORE_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
        target_dir_override = str((RESTORE_DEFAULT_DIR).resolve())
        conflict_policy = "overwrite"

    from .replay import _upstream_gen_for_sink

    upstream_gen = upstream or _upstream_gen_for_sink(run_path, sh)
    handler = bindings.RESTORE_HANDLERS[fqn]
    out_paths = handler(
        run_path=run_path,
        params=st["params"],
        realized={},
        upstream_gen=upstream_gen,
        target_override=target_dir_override,
        on_conflict=conflict_policy,
        expected_out_hashes=expected_out_hashes or None,
        record_materialization=(not ephemeral_only),
    )

    print("Replayed shards:")
    for p in out_paths:
        print(f"  {p}")

    # Track produced hashes → paths
    # If outputs were recorded, map them 1:1; else compute here.
    if outs:
        for i, o in enumerate(outs):
            dh = o["data_hash"]
            if i < len(out_paths):
                produced[dh] = Path(out_paths[i])
                created_paths.append(Path(out_paths[i]))
                produced_hashes.add(dh)
    else:
        # No recorded outputs? compute and index
        for p in out_paths:
            dh = Hash().hash_file(p)
            produced[dh] = Path(p)
            created_paths.append(Path(p))
            produced_hashes.add(dh)

    return None


# ==========
# Prepare().to_tfrecord
# ==========
def _exec_prepare_to_tfrecord(
    run_path,
    sh,
    to_path,
    produced,
    created_paths,
    produced_hashes,
):
    st = store.load_step(run_path, sh)
    outs = store.load_step_outputs(run_path, sh)

    # Expected hashes (both .tfrecord and .index if recorded)
    outs_sorted = sorted(outs, key=lambda o: o.get("name", ""))
    expected = [o["data_hash"] for o in outs_sorted]

    handler = bindings.RESTORE_HANDLERS["blase.Prepare.to_tfrecord"]
    out_paths = handler(
        run_path=run_path,
        params=st["params"],
        step_hash=sh,
        realized={},
        upstream_gen=None,
        target_override=to_path,
    )

    # Map recorded hashes to produced paths; if no outs recorded, compute hashes now
    if expected:
        for i, dh in enumerate(expected):
            if i < len(out_paths):
                p = Path(out_paths[i])
                produced[dh] = p
                created_paths.append(p)
                produced_hashes.add(dh)
    else:
        # Compute and track
        from blase.utils.hashing import Hash

        for p in out_paths:
            dh = Hash().hash_file(p)
            produced[dh] = Path(p)
            created_paths.append(Path(p))
            produced_hashes.add(dh)

    return None


# ==========
# Prepare().preview_tfrecord
# ==========
def _exec_prepare_preview_tfrecord(
    run_path,
    sh,
    to_path,
    produced,
    created_paths,
    produced_hashes,
):
    st = store.load_step(run_path, sh)
    outs = store.load_step_outputs(run_path, sh)  # may be empty if nothing persisted

    # Sort for stability; keep (hash, name)
    outs_sorted = sorted(outs, key=lambda o: o.get("name", "") or "")
    expected = [(o["data_hash"], o.get("name") or "") for o in outs_sorted]

    # Decide a target directory for side effects
    if to_path:
        target_dir = Path(to_path)
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        RESTORE_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
        target_dir = (RESTORE_DEFAULT_DIR / f"preview_tfrecord_{sh[:8]}").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

    handler = bindings.RESTORE_HANDLERS["blase.Prepare.preview_tfrecord"]
    _ = handler(
        run_path=run_path,
        params=st["params"],
        step_hash=sh,
        realized={},
        upstream_gen=None,
        target_override=str(target_dir),
    )

    # If we have recorded outputs, try to map by name → file under target_dir
    if expected:
        for dh, name in expected:
            # best-effort filename resolution
            cand = target_dir / name if name and "." in name else None
            p = None
            if cand and cand.exists():
                p = cand
            else:
                # fallback: pick any file that contains the name, else first image-like
                files = sorted(
                    [p for p in target_dir.glob("*") if p.is_file()],
                    key=lambda x: x.name,
                )
                if name:
                    p = next((f for f in files if name in f.name), None)
                if p is None and files:
                    p = files[0]

            if p and p.exists():
                produced[dh] = p
                created_paths.append(p)
                produced_hashes.add(dh)
        return None

    # No recorded outputs: hash every produced file and record those
    files = [p for p in sorted(target_dir.glob("*")) if p.is_file()]
    for p in files:
        dh = Hash().hash_file(p)
        produced[dh] = p
        created_paths.append(p)
        produced_hashes.add(dh)

    return None


# ==========
# Prepare().write_label_sidecars
# ==========
def _exec_prepare_write_label_sidecars(
    run_path,
    sh,
    to_path,
    produced,
    created_paths,
    produced_hashes,
):
    st = store.load_step(run_path, sh)
    outs = store.load_step_outputs(run_path, sh)

    outs_sorted = sorted(outs, key=lambda o: o.get("name", ""))
    expected = [o["data_hash"] for o in outs_sorted]

    handler = bindings.RESTORE_HANDLERS["blase.Prepare.write_label_sidecars"]
    out_paths = handler(
        run_path=run_path,
        params=st["params"],
        step_hash=sh,
        realized={},
        upstream_gen=None,
        target_override=to_path,
    )

    produced_pairs = []
    for p in out_paths:
        h = Hash().hash_file(p)
        produced_pairs.append((h, Path(p)))

    if expected:
        exp_set = set(expected)
        got_set = {h for h, _ in produced_pairs}

        # Verify replays produced all expected hashes
        missing = exp_set - got_set
        if missing:
            raise SystemExit(
                f"restore: sidecar replay missing expected outputs: {sorted(missing)}"
            )

        # Prefer 1:1 mapping by hash
        by_hash = {h: p for h, p in produced_pairs}
        for eh in expected:
            p = by_hash[eh]
            produced[eh] = p
            created_paths.append(p)
            produced_hashes.add(eh)
    else:
        # No recorded outs; record by computed hash
        for h, p in produced_pairs:
            produced[h] = p
            created_paths.append(p)
            produced_hashes.add(h)

    return None


def _exec_prepare_class_map_io(
    run_path, sh, to_path, produced, created_paths, produced_hashes
):
    st = store.load_step(run_path, sh)
    handler = bindings.RESTORE_HANDLERS["blase.Prepare.class_map_io"]

    out_path = handler(
        run_path=run_path,
        params=st["params"],
        step_hash=sh,
        realized={},
        upstream_gen=None,
        target_override=to_path,
    )

    if not out_path:
        return None

    p = Path(out_path)
    if not (p.exists() and p.is_file()):
        return None

    hp = Hash().hash_file(p)
    produced[hp] = p
    created_paths.append(p)
    produced_hashes.add(hp)
    return None

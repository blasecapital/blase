from blase.restoring import planner


def _collect_batch_ids(ins):
    roles = {"batch_desc", "table.batch.meta", "batch", "table.batch", "batchmeta"}
    return [i["data_hash"] for i in ins if i.get("role") in roles]


def _restore_gen_for_manifest(run_path, manifest_hash, step_hash, store, bindings):
    """
    Build and run a restore generator for a recorded manifest.

    The function inspects the upstream step to collect batch-identifiers, then
    prefers a Parquet restore path if available, falling back to the images
    restore path. It returns the generator produced by the selected bindings
    call.

    Parameters
    ----------
    run_path : pathlib.Path or str
        Path to the run directory containing `nodes/nodes.db` and `cas/`.
    manifest_hash : str
        Data hash of the recorded table/images manifest to restore.
    step_hash : str
        Upstream step hash whose inputs encode batch descriptors.
    store : object
        Store API with `load_step_inputs(...)` and optionally
        `read_parquet_params_for_manifest(...)` / `read_images_params_for_manifest(...)`.
    bindings : object
        Bindings API exposing `run_read_parquet_restore(...)` and
        `run_read_images_restore(...)`.

    Returns
    -------
    Iterator
        Generator yielding 3-tuples ``(data, meta, is_last)`` where:
        * `data` : backend-specific payload (e.g., tables or image batches)
        * `meta` : dict with provenance (e.g., ordinal, batch descriptors)
        * `is_last` : bool flagging the final batch

    Raises
    ------
    SystemExit
        If neither Parquet nor images restore parameters can be resolved
        for the given `manifest_hash`.

    Notes
    -----
    Parquet restore is attempted first. If the Parquet bindings raise an
    exception, the function silently falls back to the images path.
    """
    ts_before = planner._step_row(planner._db(run_path), step_hash)["ts_start"]
    ins = store.load_step_inputs(run_path, step_hash)

    batch_ids = _collect_batch_ids(ins)

    # parquet path
    rp_fn = getattr(store, "read_parquet_params_for_manifest", None)
    if callable(rp_fn):
        rp = rp_fn(run_path, manifest_hash, ts_before=ts_before)
        if rp:
            realized = {
                "sources": rp.get("sources"),
                "source": rp.get("source") or rp.get("directory"),
                "manifest": manifest_hash,
                "batch_descs": batch_ids,
            }
            try:
                return bindings.run_read_parquet_restore(
                    run_path=run_path, params=rp, realized=realized, transform_fn=None
                )
            except Exception:
                pass  # fall back to images

    # images path
    ri_fn = getattr(store, "read_images_params_for_manifest", None)
    if callable(ri_fn):
        ri = ri_fn(run_path, manifest_hash, ts_before=ts_before)
        if ri:
            realized = {
                "source": ri.get("directory"),
                "manifest": manifest_hash,
                "batch_descs": batch_ids,
            }
            return bindings.run_read_images_restore(
                run_path=run_path, params=ri, realized=realized, transform_fn=None
            )

    raise SystemExit(
        "replay: cannot find read_parquet or read_images params for manifest"
    )

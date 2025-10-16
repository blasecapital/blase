from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List, Set

from blase.cli.restore.replay_cas import (
    _assert_path_matches_hash,
)
from blase.cli.restore.replay import (
    _exec_plan_for_step,
    _build_stream_for_step,
    _is_stream_fqn,
    _upstream_gen_for_sink,
)
from blase.restoring import store, materialize, bindings
from blase.utils.config import RESTORE_DEFAULT_DIR
from blase.restoring.materialize import NeedReplay
from blase.restore import step as restore_step


class Policy(SimpleNamespace):
    on_conflict: str
    keep_intermediates: bool = False
    limit_batches: Optional[int] = None
    backend: Optional[str] = None
    mode: Optional[str] = None
    to: Optional[str] = None


def run_data(
    run_path: Path, data_hash: str, mode: str, to: Optional[str], policy: Policy
) -> int:
    kind = store.get_data_kind(run_path, data_hash) or "data"
    # try fast materialize first
    try:
        if to:
            to_path = Path(to).resolve()
            out = materialize.ensure_local(
                run_path,
                data_hash,
                kind=kind,
                to_dir=to_path.parent,
                target_name=to_path.name,
                on_conflict=policy.on_conflict,
            )
        else:
            RESTORE_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
            ext = "csv" if kind == "csv" else "bin"
            out = materialize.ensure_local(
                run_path,
                data_hash,
                kind=kind,
                to_dir=RESTORE_DEFAULT_DIR,
                target_name=f"{data_hash}.{ext}",
                on_conflict=policy.on_conflict,
            )
        _assert_path_matches_hash(Path(out), data_hash, "fast-path materialize")
        print(f"Materialized: {out}")
        return 0
    except NeedReplay:
        pass  # fall through to replay

    # Replay path
    seen_steps: Set[str] = set()
    created_paths: List[Path] = []

    # find the producer step and execute its forward plan
    prod = store.producer_step_for_data(run_path, data_hash)
    if not prod:
        raise SystemExit(f"[restore] no producer step recorded for data {data_hash}")

    produced = _exec_plan_for_step(
        run_path,
        prod,
        to_path=to,
        backend_override=getattr(policy, "backend", None),
        seen_steps=seen_steps,
        created_paths=created_paths,
    )

    # ---- VERIFY FINAL ARTIFACT MATCHES REQUESTED HASH ----
    # 1) Prefer the actual path that produced THIS hash (robust against renames)
    final_path = produced.get(data_hash)

    # 2) Fallbacks if mapping isn’t available (shouldn’t happen, but be safe)
    if final_path is None:
        if to:
            final_path = Path(to).resolve()
        else:
            # fallback to sink target basename under RESTORE_DEFAULT_DIR
            sink_step = store.producer_step_for_data(run_path, data_hash)
            if not sink_step:
                raise SystemExit(
                    "restore: cannot locate producing sink for final verification"
                )
            sink_meta = store.load_step(run_path, sink_step)
            default_name = Path(
                sink_meta["params"].get("target", f"{data_hash}.csv")
            ).name
            final_path = (RESTORE_DEFAULT_DIR / default_name).resolve()

    if not final_path.exists():
        raise SystemExit(f"restore: expected final output does not exist: {final_path}")

    _assert_path_matches_hash(final_path, data_hash, "final output")
    print(f"Verified: {final_path} == {data_hash[:16]}…")

    # --- Cleanup intermediates unless user asked to keep them ---
    if not getattr(policy, "keep_intermediates", False):
        final_real = final_path.resolve()

        # De-dupe by resolved path and remove everything except the final
        seen: set[Path] = set()
        for p in created_paths:
            try:
                r = p.resolve()
            except Exception:
                continue
            if r in seen:
                continue
            seen.add(r)
            if r == final_real:
                continue
            try:
                Path(r).unlink(missing_ok=True)
            except Exception:
                # best-effort: ignore failures (e.g., already deleted)
                pass

        return 0

    # Non-sink producer: stream verify only for now
    if mode in ("verify",):
        gen = (
            _build_stream_for_step(run_path, prod)
            if _is_stream_fqn(store.load_step(run_path, prod)["function_fqn"])
            else restore_step(run_path, prod, kind="data")
        )
        total = 0
        for i, (b, last) in enumerate(gen, 1):
            n = len(b) if hasattr(b, "__len__") else "?"
            print(f"[verify] batch {i}: {n} rows  last={last}")
            if isinstance(n, int):
                total += n
            if getattr(policy, "limit_batches", None) and i >= policy.limit_batches:
                break
        print(f"[verify] total rows (best-effort): {total}")
        return 0

    raise SystemExit(
        "restore: replay for non-sink data is not implemented yet in --data mode. "
        "Use `--step <producer_step>` or `--mode verify`."
    )


def run_step(
    run_path: Path, step_hash: str, mode: str, to: Optional[str], policy: Policy
) -> int:
    st = store.load_step(run_path, step_hash)
    fqn = st["function_fqn"]

    # Wrapper for "verify" and "replay"
    limit_batches = getattr(policy, "limit_batches", None)

    # ---- MATERIALIZE: just bring recorded outputs back (no replay) ----
    if policy.mode == "materialize":
        outs = store.load_step_outputs(run_path, step_hash)
        if not outs:
            print("No outputs recorded for this step.")
            return 2

        # Take the first output
        out_hash = outs[0]["data_hash"]
        kind = store.get_data_kind(run_path, out_hash) or "data"

        # Resolve target path pieces
        to_path = Path(policy.to).resolve()
        to_dir = to_path.parent
        target_name = to_path.name

        try:
            path = materialize.ensure_local(
                run_path,
                out_hash,
                kind=kind,
                to_dir=to_dir,
                target_name=target_name,
                on_conflict=policy.on_conflict,
            )
            print(f"Materialized: {path}")
            return 0
        except materialize.NeedReplay:
            print("Output not available locally. Use:")
            print(f"  blase restore plan --step {step_hash}")
            print(f"  blase restore run  --step {step_hash} --mode replay")
            return 3

    # ---- VERIFY: stream results without writing ----
    if policy.mode == "verify":
        if fqn in ("blase.Load.save_to_csv", "blase.Load.save_images_to_parquet"):
            upstream_gen = _upstream_gen_for_sink(run_path, step_hash)
            gen = upstream_gen
        elif _is_stream_fqn(fqn):
            gen = _build_stream_for_step(run_path, step_hash)
        else:
            gen = restore_step(run_path, step_hash, kind="data")

        def _coerce_stream_2(gen):
            for item in gen:
                if isinstance(item, tuple):
                    if len(item) == 3:
                        b, _meta, last = item
                    elif len(item) == 2:
                        b, last = item
                    else:
                        b, last = item, False
                else:
                    b, last = item, False
                yield b, last

        total = 0
        for i, (b, last) in enumerate(_coerce_stream_2(gen), 1):
            try:
                n = len(b)
            except Exception:
                n = "?"
            print(f"[verify] batch {i}: {n} rows  last={last}")
            if isinstance(n, int):
                total += n
            if limit_batches and i >= limit_batches:
                break
        print(f"[verify] total rows (best-effort): {total}")
        return 0

    # ---- REPLAY: for sinks, wire upstream; for non-sinks, just restore ----
    if policy.mode == "replay":
        if fqn in (
            "blase.Load.save_to_csv",
            "blase.Load.save_images_to_parquet",
            "blase.Examine.preview_images",
            "blase.Prepare.compute_stats",
            "blase.Prepare.build_manifest",
            "blase.Prepare.split",
            "blase.Prepare.to_tfrecord",
            "blase.Prepare.preview_tfrecord",
        ):
            upstream_gen = (
                _upstream_gen_for_sink(run_path, step_hash)
                if fqn.startswith("blase.Load.")
                else None
            )
            handler = bindings.RESTORE_HANDLERS[fqn]
            if fqn.startswith("blase.Examine") or fqn.startswith("blase.Prepare"):
                out_path = handler(
                    run_path=run_path,
                    params=st["params"],
                    step_hash=step_hash,
                    realized={},  # sinks don’t need extra realized inputs
                    upstream_gen=upstream_gen,
                    target_override=getattr(policy, "to", None),
                    backend_override=getattr(policy, "backend", None),
                )
            else:
                out_path = handler(
                    run_path=run_path,
                    params=st["params"],
                    realized={},  # sinks don’t need extra realized inputs
                    upstream_gen=upstream_gen,
                    target_override=getattr(policy, "to", None),
                    backend_override=getattr(policy, "backend", None),
                )
            print(out_path)
            return 0

        # non-sinks: as before
        gen = (
            _build_stream_for_step(run_path, step_hash)
            if _is_stream_fqn(fqn)
            else restore_step(run_path, step_hash, kind="data")
        )
        for i, (b, last) in enumerate(gen, 1):
            n = len(b) if hasattr(b, "__len__") else "?"
            print(f"batch {i}: {n} items, last={last}")
            if getattr(policy, "limit_batches", None) and i >= policy.limit_batches:
                break
        return 0

    raise SystemExit(f"[restore] unknown mode: {policy.mode}")

def _print_rows(rows):
    for r in rows:
        print(" | ".join(str(v) for v in r))


def _print_step(con, step_hash: str):
    s = con.execute("SELECT * FROM steps WHERE step_hash=?", (step_hash,)).fetchone()
    if not s:
        print(f"[restore] step not found: {step_hash}")
        return
    print(f"step: {s['step_hash']}  fqn={s['function_fqn']}  status={s['status']}")
    print(f"  run_id={s['run_id']}  ts_start={s['ts_start']}  ts_end={s['ts_end']}")
    print(f"  params={s['params_json']}")
    ins = con.execute(
        "SELECT data_hash, role, arg_name FROM step_inputs WHERE step_hash=?",
        (step_hash,),
    ).fetchall()
    print("  inputs:")
    for r in ins:
        print(f"    - {r['role']:8s} {r['data_hash']} arg={r['arg_name']}")
    outs = con.execute(
        "SELECT data_hash, name FROM step_outputs WHERE step_hash=?", (step_hash,)
    ).fetchall()
    print("  outputs:")
    for r in outs:
        print(f"    - {r['name']:8s} {r['data_hash']}")

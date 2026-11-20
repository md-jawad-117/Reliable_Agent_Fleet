"""Batch experiments: many random fault schedules x message-loss rates, both systems.

Every scenario (seed, #faults, loss) is run once with the baseline and once with
our runtime, with the *same* fault schedule and network randomness.

    python experiments.py --seeds 30                # full sweep -> results/runs.csv
    python experiments.py --pilot                   # quick stress test, prints anomalies
"""
import argparse
import csv
import gc
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from fleet.net import random_faults
from fleet.sim import Simulation

N_FAULTS = [0, 1, 2, 3, 4]
LOSS = [0.0, 0.05, 0.15]


def run_one(args):
    try:
        return _run_one(args)
    except Exception as e:
        mode, seed, n_faults, loss = args
        return dict(mode=mode, seed=seed, n_faults=n_faults, loss=loss, error=f"{type(e).__name__}: {e}")
    finally:
        # A Simulation holds reference cycles that pin ~165 MB of MuJoCo memory each; Python's
        # GC does not see that C-side memory, so without this a worker grows by GBs and the
        # machine runs out of memory part-way through a sweep.
        gc.collect()


PLANNER = {"kind": "scripted", "model": None}     # set per worker by _init_worker


def _make_planner():
    kind = PLANNER["kind"]
    if kind == "scripted":
        return None
    from fleet.llm_planner import DEFAULT_MODEL, LLMPlanner
    if kind == "fake":
        from fleet.fake_agent import ImpatientAgentClient
        return LLMPlanner(client=ImpatientAgentClient())
    return LLMPlanner(model=PLANNER["model"] or DEFAULT_MODEL)


PLANNER_FIELDS = ["planner", "agent_calls", "agent_intents", "intent_unsafe_redo", "refused_unsafe_redo",
                  "intent_repeat", "refused_repeat", "agent_refused", "input_tokens", "output_tokens",
                  "api_seconds"]


def _run_one(args):
    mode, seed, n_faults, loss = args
    rng = np.random.default_rng(10_000 + seed * 97 + n_faults * 13 + int(loss * 1000))
    faults = random_faults(rng, n_faults)
    planner = _make_planner()
    sim = Simulation(mode, faults=faults, seed=seed, loss=loss, t_max=150.0, planner=planner)
    try:
        m = sim.run().as_dict()
        m["error"] = ""
    except Exception as e:  # keep the sweep going; record the failure
        m = sim.metrics().as_dict()
        m["error"] = f"{type(e).__name__}: {e}"
    m.update(mode=mode, seed=seed, n_faults=n_faults, loss=loss,
             finished=sim.finished_at is not None,
             faults_fired=sum(f.fired_at is not None for f in faults),
             fault_desc="; ".join(f"{f.kind}@arm{f.arm}/{f.phase}/{f.after:.0f}s/{f.duration:.0f}s"
                                  for f in faults))
    m["planner"] = PLANNER["kind"] if planner is None or PLANNER["kind"] != "claude" else planner.model
    if planner is not None:
        s = planner.stats
        m.update(agent_calls=s["calls"], agent_intents=s["intents"],
                 intent_unsafe_redo=s["intent_unsafe_redo"], refused_unsafe_redo=s["refused_unsafe_redo"],
                 intent_repeat=s["intent_repeat"], refused_repeat=s["refused_repeat"],
                 agent_refused=s["refused"], input_tokens=s["input_tokens"], output_tokens=s["output_tokens"],
                 api_seconds=round(s["api_seconds"], 1))
        if PLANNER["kind"] == "claude":
            m["decisions"] = planner.decisions
    if mode == "reliable" and (m["duplicate_shipments"] or m["collisions"] or m["orders_ok"] < 9
                               or not m["finished"]):
        m["log"] = "\n".join(f"{t:6.2f} {l}" for t, l in sim.log_lines if "start O" not in l)
    return m


FIELDS = ["mode", "seed", "n_faults", "loss", "orders_total", "orders_ok", "duplicate_shipments",
          "wrong_color", "orders_failed", "collisions", "makespan", "msgs_sent", "msgs_dropped",
          "finished", "faults_fired", "fault_desc", "error"] + PLANNER_FIELDS


def _init_worker(kind="scripted", model=None):
    PLANNER.update(kind=kind, model=model)
    _low_priority()


def _low_priority():
    """Run workers at below-normal priority so the computer stays responsive."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        else:
            os.nice(10)
    except Exception:
        pass


def _key(mode, seed, n_faults, loss):
    return (mode, int(seed), int(n_faults), round(float(loss), 4))


def _read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("seed", "n_faults", "orders_total", "orders_ok", "duplicate_shipments", "wrong_color",
                  "orders_failed", "collisions", "msgs_sent", "msgs_dropped", "faults_fired"):
            r[k] = int(r[k]) if r.get(k) not in (None, "") else 0
        r["loss"] = float(r["loss"])
        r["makespan"] = float(r["makespan"] or 0)
        r["finished"] = r["finished"] == "True"
        for k in PLANNER_FIELDS[1:]:
            v = r.get(k)
            r[k] = float(v) if v not in (None, "") else 0
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", default=os.path.join("results", "runs.csv"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--fresh", action="store_true", help="ignore existing results and start over")
    ap.add_argument("--planner", choices=["scripted", "fake", "claude"], default="scripted",
                    help="who assigns orders: the scripted loop, an offline impatient agent, or Claude")
    ap.add_argument("--model", default=None, help="Claude model id (default: llm_planner.DEFAULT_MODEL)")
    ap.add_argument("--faults", type=int, nargs="*", default=None, help="fault counts (default 0-4)")
    ap.add_argument("--loss", type=float, nargs="*", default=None, help="loss rates (default 0 .05 .15)")
    args = ap.parse_args()

    n_faults = args.faults if args.faults is not None else N_FAULTS
    losses = args.loss if args.loss is not None else LOSS
    if args.pilot:
        jobs = [(mode, s, 3, 0.15) for s in range(args.seeds) for mode in ("baseline", "reliable")]
        out = os.path.join("results", "pilot.csv" if args.planner == "scripted" else f"pilot_{args.planner}.csv")
    else:
        jobs = [(mode, s, k, l) for s, k, l in itertools.product(range(args.seeds), n_faults, losses)
                for mode in ("baseline", "reliable")]
        out = args.out if args.planner == "scripted" else args.out.replace(".csv", f"_{args.planner}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # ---- resume: skip runs already saved
    done = set()
    if os.path.exists(out) and not args.fresh and not args.pilot:
        for r in _read_csv(out):
            if not r["error"]:
                done.add(_key(r["mode"], r["seed"], r["n_faults"], r["loss"]))
    else:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    todo = [j for j in jobs if _key(*j) not in done]
    print(f"{len(jobs)} runs total, {len(done)} already saved, {len(todo)} to go, "
          f"{args.workers} workers", flush=True)

    t0 = time.time()
    anomalies = []
    decisions_path = out.replace(".csv", "_decisions.jsonl")
    with Pool(args.workers, initializer=_init_worker, initargs=(args.planner, args.model),
              maxtasksperchild=20) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, todo, chunksize=1)):
            if "log" in r:
                anomalies.append(r)
            if r.get("decisions"):                       # audit trail of every agent tool call
                with open(decisions_path, "a") as f:
                    f.write(json.dumps({k: r[k] for k in ("mode", "seed", "n_faults", "loss", "decisions")}) + "\n")
            with open(out, "a", newline="") as f:            # save every run immediately
                csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore").writerow(r)
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                eta = el / (i + 1) * (len(todo) - i - 1)
                print(f"  {i + 1}/{len(todo)}  elapsed {el / 60:.1f} min, ~{eta / 60:.1f} min left", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)

    rows = _read_csv(out)
    errs = [r for r in rows if r["error"]]
    print(f"runs with engine errors: {len(errs)}")
    for r in errs[:5]:
        print("  ", r["mode"], r["seed"], r["n_faults"], r["loss"], r["error"])
    rows = [r for r in rows if not r["error"]]
    bad = [r for r in anomalies if r["duplicate_shipments"] or r["collisions"]]
    print(f"reliable runs with a duplicate or collision this session: {len(bad)}")
    for r in bad[:5]:
        print("\n=== reliable anomaly", {k: r[k] for k in ("seed", "n_faults", "loss", "orders_ok",
                                                         "duplicate_shipments", "collisions", "finished")})
        print(r["fault_desc"])
        print(r["log"])
    for mode in ("baseline", "reliable"):
        rs = [r for r in rows if r["mode"] == mode]
        print(f"{mode:9s} ok={np.mean([r['orders_ok'] for r in rs]):.2f}/9 "
              f"dup={np.mean([r['duplicate_shipments'] for r in rs]):.2f} "
              f"coll={np.mean([r['collisions'] for r in rs]):.2f} "
              f"unsafe_runs={sum(1 for r in rs if r['duplicate_shipments'] or r['collisions'])}/{len(rs)} "
              f"unfinished={sum(not r['finished'] for r in rs)} "
              f"makespan={np.mean([r['makespan'] for r in rs]):.1f}")
        if args.planner != "scripted":
            print(f"          agent: calls={sum(r['agent_calls'] for r in rs):.0f} "
                  f"intents={sum(r['agent_intents'] for r in rs):.0f} "
                  f"unsafe_redo={sum(r['intent_unsafe_redo'] for r in rs):.0f} "
                  f"(refused {sum(r['refused_unsafe_redo'] for r in rs):.0f}) "
                  f"repeat={sum(r['intent_repeat'] for r in rs):.0f} "
                  f"tokens in/out={sum(r['input_tokens'] for r in rs):.0f}/{sum(r['output_tokens'] for r in rs):.0f}")
    print("results in", out, flush=True)


if __name__ == "__main__":
    main()

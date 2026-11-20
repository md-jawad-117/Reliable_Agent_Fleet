"""Agent fault injection experiments: what happens when the *planner* misbehaves?

Two studies, both run with the same seeded agent behaviour against both runtimes:

  A. personalities - each agent fault on its own (rate 0.3) and all of them together,
     with and without robot/network faults:   python agent_experiments.py --study A
  B. dose-response - the "mixed" agent at error rates 0 .. 0.4, with 2 robot faults:
                                               python agent_experiments.py --study B

Results go to results/agents.csv (resumable, like experiments.py). No API, no cost.
"""
import argparse
import csv
import gc
import itertools
import os
import time
import zlib
from multiprocessing import Pool

import numpy as np

from experiments import _low_priority
from fleet.fake_agent import FaultyAgentClient
from fleet.llm_planner import LLMPlanner
from fleet.net import random_faults
from fleet.sim import Simulation

OUT = os.path.join("results", "agents.csv")
LOSS = 0.05
FIELDS = ["study", "agent", "rate", "mode", "seed", "n_faults", "loss", "orders_ok", "duplicate_shipments",
          "collisions", "orders_failed", "makespan", "finished", "agent_intents", "intent_normal",
          "intent_unsafe_redo", "refused_unsafe_redo", "intent_repeat", "refused_repeat",
          "intent_wrong_arm", "refused_wrong_arm", "intent_invalid", "agent_refused", "error"]


def jobs(study, seeds):
    out = []
    if study in ("A", "all"):
        for agent, s, k in itertools.product(["impatient", "forgetful", "confused", "hallucinating", "mixed"],
                                             range(seeds), [0, 2]):
            out += [("A", agent, 0.3, mode, s, k) for mode in ("baseline", "reliable")]
    if study in ("B", "all"):
        for rate, s in itertools.product([0.0, 0.05, 0.1, 0.2, 0.4], range(seeds)):
            out += [("B", "mixed", rate, mode, s, 2) for mode in ("baseline", "reliable")]
    return out


def run_one(job):
    study, agent, rate, mode, seed, n_faults = job
    row = dict(study=study, agent=agent, rate=rate, mode=mode, seed=seed, n_faults=n_faults, loss=LOSS)
    try:
        # same robot faults and the same agent random stream for both runtimes
        rng = np.random.default_rng(20_000 + seed * 97 + n_faults * 13)
        faults = random_faults(rng, n_faults)
        agent_seed = zlib.crc32(f"{agent}-{rate}-{seed}-{n_faults}".encode())
        planner = LLMPlanner(client=FaultyAgentClient(agent, rate=rate, seed=agent_seed), max_calls=150)
        sim = Simulation(mode, faults=faults, seed=seed, loss=LOSS, t_max=150.0, planner=planner)
        m = sim.run()
        s = planner.stats
        row.update(orders_ok=m.orders_ok, duplicate_shipments=m.duplicate_shipments, collisions=m.collisions,
                   orders_failed=m.orders_failed, makespan=m.makespan, finished=sim.finished_at is not None,
                   agent_intents=s["intents"], agent_refused=s["refused"], error="",
                   **{k: s[k] for k in FIELDS if k.startswith(("intent_", "refused_"))})
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    finally:
        gc.collect()
    return row


def key(r):
    return (r["study"], r["agent"], round(float(r["rate"]), 3), r["mode"], int(r["seed"]), int(r["n_faults"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["A", "B", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    done = set()
    if os.path.exists(OUT) and not args.fresh:
        with open(OUT, newline="") as f:
            done = {key(r) for r in csv.DictReader(f) if not r["error"]}
    else:
        with open(OUT, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    todo = [j for j in jobs(args.study, args.seeds)
            if key(dict(zip(("study", "agent", "rate", "mode", "seed", "n_faults"), j))) not in done]
    print(f"{len(todo)} runs to go ({len(done)} saved), {args.workers} workers", flush=True)

    t0 = time.time()
    with Pool(args.workers, initializer=_low_priority, maxtasksperchild=20) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, todo, chunksize=1)):
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore").writerow(r)
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                print(f"  {i + 1}/{len(todo)}  {el / 60:.1f} min, ~{el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min left",
                      flush=True)
    print("results in", OUT, flush=True)


if __name__ == "__main__":
    main()

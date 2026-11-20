"""Ablation study: which mechanism of our runtime prevents which failure?

Each variant switches off one mechanism (or a pair, to expose mechanisms that back each
other up) and replaces it with the simpler behaviour it guards against:

  no_fencing     robots ignore tokens and REVOKE; a rebooted robot resumes its job unasked
  no_leases      no heartbeat-renewed leases: robots never check ownership before acting,
                 the coordinator uses fixed timeouts instead
  no_camera      no physical reconciliation: an expired job is redone blindly, an expired
                 airspace reservation is freed without looking
  no_dedup       robots re-execute a command they have already taken (same order/token)
  no_done_retry  DONE is sent once instead of repeated until acknowledged

Two complementary tests, both against every variant (plus the baseline):

  targeted - one scripted fault per failure mode, 10 network seeds each
  random   - the random-fault grid of experiments.py (smaller), to measure real frequency

    python ablation_experiments.py             -> results/ablations.csv
    python plot_ablations.py                   -> figures + table
"""
import argparse
import csv
import gc
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np

from experiments import _low_priority
from fleet.net import Fault, random_faults
from fleet.sim import Simulation

OUT = os.path.join("results", "ablations.csv")

VARIANTS = ["full", "no_fencing", "no_leases", "no_camera", "no_dedup", "no_done_retry",
            "no_fencing+no_leases", "no_dedup+no_leases", "no_done_retry+no_camera", "baseline"]

# one scripted fault per failure mode (fires when the arm first reaches `phase` after `after` s)
SCENARIOS = {
    "zombie":       ("crash", 0, "REACH", 20.0,
                     "arm crashes mid-reach and stays down long enough for its job to be reassigned"),
    "crash_in_air": ("crash", 1, "DESCEND", 15.0,
                     "arm crashes while reaching into the table and is frozen inside the airspace"),
    "partition":    ("partition", 2, "DESCEND", 12.0,
                     "arm is cut off from the coordinator mid-reach"),
    "lost_done":    ("mute", 1, "OPEN", 4.0,
                     "the arm's DONE (and heartbeats) are lost right after it ships"),
    "dup_command":  ("mute", 0, "GO_STAGE", 3.0,
                     "the arm's ACKs are lost, so the coordinator keeps re-sending the command"),
}

FIELDS = ["test", "scenario", "variant", "seed", "n_faults", "loss", "orders_ok", "duplicate_shipments",
          "collisions", "orders_failed", "makespan", "finished", "error"]


def jobs(seeds_targeted, seeds_random):
    out = []
    for sc, v, s in itertools.product(SCENARIOS, VARIANTS, range(seeds_targeted)):
        out.append(("targeted", sc, v, s, 1, 0.05))
    for v, s, k, l in itertools.product(VARIANTS, range(seeds_random), [2, 4], [0.05, 0.15]):
        out.append(("random", "random", v, s, k, l))
    return out


def run_one(job):
    test, scenario, variant, seed, n_faults, loss = job
    row = dict(test=test, scenario=scenario, variant=variant, seed=seed, n_faults=n_faults, loss=loss)
    try:
        if test == "targeted":
            kind, arm, phase, dur, _ = SCENARIOS[scenario]
            faults = [Fault(kind, arm, after=1.0, duration=dur, phase=phase, label=scenario)]
        else:
            rng = np.random.default_rng(30_000 + seed * 97 + n_faults * 13 + int(loss * 1000))
            faults = random_faults(rng, n_faults)
        mode = "baseline" if variant == "baseline" else "reliable"
        ablate = [] if variant in ("full", "baseline") else variant.split("+")
        sim = Simulation(mode, faults=faults, seed=seed, loss=loss, t_max=150.0, ablate=ablate)
        m = sim.run()
        row.update(orders_ok=m.orders_ok, duplicate_shipments=m.duplicate_shipments, collisions=m.collisions,
                   orders_failed=m.orders_failed, makespan=m.makespan, finished=sim.finished_at is not None,
                   error="")
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    finally:
        gc.collect()
    return row


def key(r):
    return (r["test"], r["scenario"], r["variant"], int(r["seed"]), int(r["n_faults"]), round(float(r["loss"]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-targeted", type=int, default=10)
    ap.add_argument("--seeds-random", type=int, default=30)
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
    todo = [j for j in jobs(args.seeds_targeted, args.seeds_random)
            if key(dict(zip(("test", "scenario", "variant", "seed", "n_faults", "loss"), j))) not in done]
    print(f"{len(todo)} runs to go ({len(done)} saved), {args.workers} workers", flush=True)

    t0 = time.time()
    with Pool(args.workers, initializer=_low_priority, maxtasksperchild=20) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, todo, chunksize=1)):
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore").writerow(r)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                print(f"  {i + 1}/{len(todo)}  {el / 60:.1f} min, ~{el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min left",
                      flush=True)
    print("results in", OUT, flush=True)


if __name__ == "__main__":
    main()

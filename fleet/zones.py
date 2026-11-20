"""Fine-grained airspace: which picks can safely happen at the same time?

Instead of one lock for the whole table, an arm reserves *one pick*: (arm a, box k).
Two reservations (a, k) and (b, l) may be held at once only if the two arms' whole
pick paths (staging -> above box -> box -> lift -> staging) never come within
CLEARANCE of each other, for *any* timing of the two motions. We check that offline
with MuJoCo's own collision detector on the real Panda collision meshes, and cache it.

Because the check covers every pair of poses along both paths (not one particular
timing), a crash or delay that freezes an arm anywhere on its path is still covered.
"""
from __future__ import annotations

import json
import os

import mujoco
import numpy as np

from .scene import BOX_HALF, N_ARMS, TABLE_TOP, build_model

CLEARANCE = 0.03       # links must stay at least 3 cm apart
STEP = 0.02            # path sampling (m)
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conflicts.json")
_table = None


def _path_points(stage, box_xy):
    x, y = box_xy
    z_box = TABLE_TOP + BOX_HALF
    way = [np.array(stage), np.array([x, y, z_box + 0.14]), np.array([x, y, z_box]),
           np.array([x, y, 0.30]), np.array(stage)]
    pts = []
    for p0, p1 in zip(way[:-1], way[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / STEP)))
        pts += [p0 + (p1 - p0) * s for s in np.linspace(0, 1, n, endpoint=False)]
    # abort branch from the approach: straight up to 0.30 (the robot's abort rule)
    pts += [np.array([x, y, 0.30])]
    return pts + [way[-1]]


def compute_table(verbose=True):
    from .sim import Simulation   # reuse the exact arm setup (orientation, staging pose)
    sim = Simulation("reliable", faults=[])
    m, lay, arms = sim.model, sim.lay, sim.arms
    # collision copy with a safety margin on every arm geom
    body_arm = np.full(m.nbody, -1)
    for i in range(N_ARMS):
        for b in range(m.nbody):
            if m.body(b).name.startswith(f"arm{i}/"):
                body_arm[b] = i
    mm, _ = build_model()
    for g in range(mm.ngeom):
        if body_arm[mm.geom_bodyid[g]] >= 0:
            mm.geom_margin[g] = CLEARANCE
    dd = mujoco.MjData(mm)
    dd.qpos[:] = sim.data.qpos

    # joint-space path per (arm, box)
    paths = {}
    for a in arms:
        for k, (bx, by, _) in enumerate(lay.box_start):
            q = a.q_des.copy()
            qs = []
            for p in _path_points(lay.staging[a.idx], (bx, by)):
                q = a.solve_ik(p, q, iters=8)
                qs.append(q.copy())
            paths[(a.idx, k)] = np.array(qs)

    nb = len(lay.box_start)
    conflict = np.zeros((N_ARMS, nb, N_ARMS, nb), bool)
    for a in range(N_ARMS):
        for b in range(a + 1, N_ARMS):
            for k in range(nb):
                for l in range(nb):
                    hit = False
                    for qa in paths[(a, k)]:
                        dd.qpos[arms[a].qadr] = qa
                        for qb in paths[(b, l)]:
                            dd.qpos[arms[b].qadr] = qb
                            mujoco.mj_kinematics(mm, dd)
                            mujoco.mj_collision(mm, dd)
                            for c in dd.contact[: dd.ncon]:
                                if {body_arm[mm.geom_bodyid[c.geom1]],
                                        body_arm[mm.geom_bodyid[c.geom2]]} == {a, b}:
                                    hit = True
                                    break
                            if hit:
                                break
                        if hit:
                            break
                    conflict[a, k, b, l] = conflict[b, l, a, k] = hit
            if verbose:
                print(f"arm{a} vs arm{b}: {int(conflict[a, :, b, :].sum())}/{nb * nb} box pairs conflict",
                      flush=True)
    return conflict


def table():
    global _table
    if _table is None:
        if os.path.exists(_CACHE):
            with open(_CACHE) as f:
                _table = np.array(json.load(f), bool)
        else:
            _table = compute_table()
            with open(_CACHE, "w") as f:
                json.dump(_table.tolist(), f)
    return _table


def compatible(a, k, b, l) -> bool:
    """Can arm a pick box k while arm b picks box l?"""
    if a == b:
        return False
    return not table()[a, k, b, l]


if __name__ == "__main__":
    import time
    t0 = time.time()
    if os.path.exists(_CACHE):
        os.remove(_CACHE)
    t = table()
    print(f"done in {time.time() - t0:.0f}s; compatible fraction "
          f"{1 - t.sum() / (t.size - N_ARMS * t.shape[1] ** 2):.2f}")

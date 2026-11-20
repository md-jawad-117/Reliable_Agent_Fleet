"""The physical world: boxes, grasping, shipments, collisions and 'perception'.

Everything here is ground truth. Nodes only see it through narrow interfaces:
  - a robot can perceive boxes on the table near it (its camera),
  - the reliable coordinator can query a cell camera when it must reconcile
    an uncertain state ("is that box still on the table?", "is an arm in the
    airspace?").
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import AIRSPACE_RADIUS, BIN_HALF, BOX_HALF, TABLE_HALF, TABLE_TOP, N_ARMS

LAMP = {
    "idle": (0.25, 0.45, 0.9, 1),
    "busy": (0.1, 0.85, 0.3, 1),
    "crashed": (0.95, 0.1, 0.1, 1),
    "partitioned": (0.7, 0.2, 0.9, 1),
    "aborting": (1.0, 0.75, 0.1, 1),
    "waiting": (0.6, 0.6, 0.6, 1),
}


@dataclass
class Shipment:
    box: int
    color: str
    bin_arm: int
    order: str | None
    t: float


class World:
    def __init__(self, model, data, layout, arms):
        self.m, self.d, self.lay, self.arms = model, data, layout, arms
        self.n_box = len(layout.box_start)
        self.box_qadr = [model.jnt_qposadr[model.joint(f"box{k}_free").id] for k in range(self.n_box)]
        self.box_dadr = [model.jnt_dofadr[model.joint(f"box{k}_free").id] for k in range(self.n_box)]
        self.box_geom = [model.geom(f"box{k}_geom").id for k in range(self.n_box)]
        self.box_color = list(layout.box_color)
        self.carried_by = [None] * self.n_box          # arm index or None
        self.carry_order = [None] * self.n_box         # order the robot believed it was serving
        self.shipped = [False] * self.n_box
        self.picked_by: list = [None] * self.n_box     # (arm, t) of the last pick, seen by the cell camera
        self.now = 0.0
        self.shipments: list[Shipment] = []
        self.lamp_geom = [model.geom(f"lamp{i}").id for i in range(N_ARMS)]
        self.air_geom = model.geom("airspace").id
        # which arm owns each body (for arm-arm collision detection)
        self.body_arm = np.full(model.nbody, -1)
        for b in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name.startswith("arm"):
                self.body_arm[b] = int(name[3])
        self.arm_check_bodies = [[model.body(f"arm{i}/{n}").id
                                  for n in ("hand", "link7", "link6", "link5", "left_finger", "right_finger")]
                                 for i in range(N_ARMS)]
        self.collision_events = 0
        self._colliding_pairs: dict = {}
        self.events: list[tuple[float, str]] = []

    # ---------- queries ----------
    def box_pos(self, k) -> np.ndarray:
        a = self.box_qadr[k]
        return self.d.qpos[a:a + 3].copy()

    def box_on_table(self, k) -> bool:
        p = self.box_pos(k)
        return (self.carried_by[k] is None and abs(p[0]) < TABLE_HALF and abs(p[1]) < TABLE_HALF
                and p[2] > TABLE_TOP and p[2] < TABLE_TOP + 4 * BOX_HALF)

    def boxes_on_table(self, color=None):
        return [k for k in range(self.n_box)
                if self.box_on_table(k) and (color is None or self.box_color[k] == color)]

    def arms_in_airspace(self, margin: float = 0.0) -> list[int]:
        """Arms with their tool tip, wrist or forearm inside the shared airspace cylinder."""
        out = []
        for a in self.arms:
            pts = [a.tcp_pos()] + [self.d.xpos[b] for b in self.arm_check_bodies[a.idx]]
            for p in pts:
                if np.hypot(p[0], p[1]) < AIRSPACE_RADIUS + margin and p[2] < TABLE_TOP + 0.55:
                    out.append(a.idx)
                    break
        return out

    def shipped_since(self, bin_arm: int, t0: float):
        return [s for s in self.shipments if s.bin_arm == bin_arm and s.t >= t0]

    def camera_box_status(self, arm: int, t0: float):
        """Cell-camera view of what `arm` did with boxes since t0.

        Returns ("shipped", k) | ("held", k) | ("moving", k) | ("dropped", k) | ("none", None).
        """
        for k in range(self.n_box):
            pb = self.picked_by[k]
            if pb is None or pb[0] != arm or pb[1] < t0:
                continue
            if self.shipped[k]:
                return "shipped", k
            if self.carried_by[k] == arm:
                return "held", k
            dv = self.box_dadr[k]
            if np.linalg.norm(self.d.qvel[dv:dv + 3]) > 0.05 or self.box_pos(k)[2] > 0.06:
                return "moving", k
            return "dropped", k
        return "none", None

    # ---------- grasp ----------
    def try_grasp(self, arm: int, k: int, order: str) -> bool:
        tcp = self.arms[arm].tcp_pos()
        if not self.box_on_table(k) or np.linalg.norm(self.box_pos(k) - tcp) > 0.035:
            return False
        self.carried_by[k] = arm
        self.carry_order[k] = order
        self.picked_by[k] = (arm, self.now)
        self.m.geom_contype[self.box_geom[k]] = 0
        self.m.geom_conaffinity[self.box_geom[k]] = 0
        return True

    def release(self, arm: int):
        for k in range(self.n_box):
            if self.carried_by[k] == arm:
                self.carried_by[k] = None
                self.m.geom_contype[self.box_geom[k]] = 1
                self.m.geom_conaffinity[self.box_geom[k]] = 1
                d = self.box_dadr[k]
                self.d.qvel[d:d + 6] = 0

    def holding(self, arm: int):
        for k in range(self.n_box):
            if self.carried_by[k] == arm:
                return k
        return None

    # ---------- per-step physics hooks ----------
    def pre_step(self):
        for k in range(self.n_box):
            arm = self.carried_by[k]
            if arm is None:
                continue
            a = self.box_qadr[k]
            self.d.qpos[a:a + 3] = self.arms[arm].tcp_pos()
            self.d.qpos[a + 3:a + 7] = (1, 0, 0, 0)
            d = self.box_dadr[k]
            self.d.qvel[d:d + 6] = 0

    def post_step(self, t: float):
        self.now = t
        # shipments: a free box that has settled inside a bin
        for k in range(self.n_box):
            if self.shipped[k] or self.carried_by[k] is not None:
                continue
            p = self.box_pos(k)
            d = self.box_dadr[k]
            if np.linalg.norm(self.d.qvel[d:d + 3]) > 0.05:
                continue
            for i, (bx, by) in enumerate(self.lay.bin_pos):
                if abs(p[0] - bx) < BIN_HALF and abs(p[1] - by) < BIN_HALF and p[2] < 0.16:
                    self.shipped[k] = True
                    s = Shipment(k, self.box_color[k], i, self.carry_order[k], t)
                    self.shipments.append(s)
                    break

    def check_collisions(self, t: float):
        now = set()
        for c in self.d.contact[: self.d.ncon]:
            a1 = self.body_arm[self.m.geom_bodyid[c.geom1]]
            a2 = self.body_arm[self.m.geom_bodyid[c.geom2]]
            if a1 >= 0 and a2 >= 0 and a1 != a2:
                now.add((min(a1, a2), max(a1, a2)))
        for pair in now:
            last = self._colliding_pairs.get(pair, -10.0)
            if t - last > 1.0:
                self.collision_events += 1
                self.events.append((t, f"COLLISION arm{pair[0]} <-> arm{pair[1]}"))
            self._colliding_pairs[pair] = t
        # airspace glows red for a moment after an actual arm-arm contact
        # (several arms inside at once is now normal: they hold compatible reservations)
        inside = self.arms_in_airspace()
        recent = any(t - last < 0.5 for last in self._colliding_pairs.values())
        rgba = (1.0, 0.1, 0.1, 0.25) if recent else (0.2, 0.8, 1.0, 0.06)
        self.m.geom_rgba[self.air_geom] = rgba
        return inside

    def set_lamp(self, arm: int, state: str):
        self.m.geom_rgba[self.lamp_geom[arm]] = LAMP.get(state, LAMP["waiting"])

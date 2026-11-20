"""Glue: one deterministic simulation run = physics + network + nodes + faults + metrics."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import mujoco
import numpy as np

from .arm import ArmController
from .coordinator import BaselineCoordinator, ReliableCoordinator
from .net import Fault, Network, demo_faults
from .robot import RobotNode
from .scene import N_ARMS, build_model
from .world import World

# each switches off one mechanism of our runtime (only meaningful with mode="reliable")
ABLATIONS = {"no_fencing", "no_leases", "no_camera", "no_dedup", "no_done_retry"}

CTRL_DT = 0.01
SUBSTEPS = 5   # physics dt 0.002


@dataclass
class Metrics:
    orders_total: int
    orders_ok: int
    duplicate_shipments: int
    wrong_color: int
    orders_failed: int
    collisions: int
    makespan: float
    msgs_sent: int
    msgs_dropped: int

    def as_dict(self):
        return self.__dict__.copy()


class Simulation:
    def __init__(self, mode: str, faults: list[Fault] | None = None, seed: int = 0,
                 loss: float = 0.0, t_max: float = 90.0, planner=None, ablate=()):
        assert mode in ("baseline", "reliable")
        self.mode, self.t_max = mode, t_max
        self.model, self.lay = build_model()
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.net = Network(self.rng, loss=loss)
        scratch = mujoco.MjData(self.model)   # shared IK workspace
        self.arms = [ArmController(self.model, self.data, i, scratch) for i in range(N_ARMS)]
        for a in self.arms:
            a.reset_home()
        mujoco.mj_forward(self.model, self.data)
        for a in self.arms:
            a.capture_orientation()
            a.teleport(self.lay.staging[a.idx])
        mujoco.mj_forward(self.model, self.data)
        self.world = World(self.model, self.data, self.lay, self.arms)
        self.log_lines: list[tuple[float, str]] = []
        self.robots = [RobotNode(i, mode, self.arms[i], self.world, self.net, self.lay, self.log)
                       for i in range(N_ARMS)]
        C = BaselineCoordinator if mode == "baseline" else ReliableCoordinator
        self.coord = C(self.net, self.world, N_ARMS, self.log)
        self.coord.planner = planner         # None = scripted planner; else an LLM agent
        self.ablate = set(ablate)
        assert self.ablate <= ABLATIONS, f"unknown ablation(s): {self.ablate - ABLATIONS}"
        self.coord.ab = self.ablate
        for r in self.robots:
            r.ab = self.ablate
        self.faults = faults if faults is not None else demo_faults()
        self.t = 0.0
        self.finished_at = None
        self.active_faults: list[str] = []

    def log(self, t, msg):
        self.log_lines.append((t, msg))

    # ------------------------------------------------------------------ faults
    def _faults(self):
        t = self.t
        self.active_faults = []
        for f in self.faults:
            r = self.robots[f.arm]
            if f.fired_at is None:
                if t >= f.after and (f.phase is None or r.phase == f.phase) and not r.crashed:
                    f.fired_at = t
                    self.log(t, f"FAULT: {f.label or f.kind + ' arm' + str(f.arm)} ({f.duration:.0f}s)")
                    if f.kind == "crash":
                        r.crash(t)
                    elif f.kind == "partition":
                        self.net.blocked |= {(r.name, "coord"), ("coord", r.name)}
                    elif f.kind == "mute":
                        self.net.blocked.add((r.name, "coord"))
            elif not f.healed:
                if t >= f.fired_at + f.duration:
                    f.healed = True
                    self.log(t, f"HEAL: {f.kind} on arm{f.arm} ends")
                    if f.kind == "crash":
                        r.restart(t)
                    elif f.kind == "partition":
                        self.net.blocked -= {(r.name, "coord"), ("coord", r.name)}
                    elif f.kind == "mute":
                        self.net.blocked.discard((r.name, "coord"))
                else:
                    self.active_faults.append(f"{f.kind} arm{f.arm} ({f.fired_at + f.duration - t:.0f}s)")

    # ------------------------------------------------------------------ step
    def step(self):
        t = self.t
        self._faults()
        for dst, src, msg in self.net.deliver(t):
            if dst == "coord":
                self.coord.on_msg(t, src, msg)
            else:
                self.robots[int(dst[3])].on_msg(t, msg)
        self.coord.tick(t)
        for r in self.robots:
            r.tick(t)
        for a in self.arms:
            a.update(t)
        for _ in range(SUBSTEPS):
            self.world.pre_step()
            mujoco.mj_step(self.model, self.data)
        self.world.pre_step()
        mujoco.mj_forward(self.model, self.data)
        self.t = t + CTRL_DT
        self.world.post_step(self.t)
        self.world.check_collisions(self.t)
        self.log_lines.extend(self.world.events)
        self.world.events.clear()
        for r in self.robots:
            state = r.status
            if not r.crashed and any(f.fired_at is not None and not f.healed and f.arm == r.idx
                                     and f.kind in ("partition", "mute") for f in self.faults):
                state = "partitioned"
            self.world.set_lamp(r.idx, state)
        if self.finished_at is None and self._quiescent():
            self.finished_at = self.t

    def _quiescent(self):
        robots_idle = all(r.phase == "IDLE" and not r.queue and not r.crashed for r in self.robots)
        faults_done = all(f.healed or f.fired_at is None for f in self.faults)
        boxes_settled = all(c is None for c in self.world.carried_by)
        return self.coord.all_settled() and robots_idle and faults_done and boxes_settled

    def done(self):
        return self.t >= self.t_max or (self.finished_at is not None and self.t > self.finished_at + 1.5)

    def run(self):
        while not self.done():
            self.step()
        return self.metrics()

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> Metrics:
        per_order = Counter()
        ok, wrong = set(), 0
        colors = {o.oid: o.color for o in self.coord.orders}
        for s in self.world.shipments:
            if s.order is None:
                continue
            per_order[s.order] += 1
            if s.color != colors[s.order]:
                wrong += 1
            elif s.order not in ok:
                ok.add(s.order)
        dup = sum(max(0, n - 1) for n in per_order.values())
        return Metrics(
            orders_total=len(self.coord.orders), orders_ok=len(ok), duplicate_shipments=dup,
            wrong_color=wrong, orders_failed=len(self.coord.orders) - len(ok),
            collisions=self.world.collision_events,
            makespan=round(self.finished_at or self.t, 1),
            msgs_sent=self.net.sent, msgs_dropped=self.net.dropped)

    def hud(self) -> list[str]:
        m = self.metrics()
        stock = len(self.world.boxes_on_table())
        lines = [
            f"{'BASELINE (retry + timeouts)' if self.mode == 'baseline' else 'RELIABLE (leases + fencing + reconcile)'}",
            f"t = {self.t:5.1f} s",
            f"orders shipped correctly: {m.orders_ok}/{m.orders_total}",
            f"duplicate shipments:      {m.duplicate_shipments}",
            f"arm-arm collisions:       {m.collisions}",
            f"boxes left on table:      {stock}",
        ]
        if self.active_faults:
            lines.append("ACTIVE FAULT: " + ", ".join(self.active_faults))
        return lines

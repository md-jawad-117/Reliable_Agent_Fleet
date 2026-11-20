"""Robot node: the software running on each arm's controller.

Two behaviours share one state machine:

  baseline  - what a typical agent/robot stack does today: accept commands,
              ack them, execute, report DONE once. Durable command queue so a
              rebooted robot resumes its job. No leases, no fencing.

  reliable  - our runtime. Every command carries a fencing token. The robot
              holds a *lease* on its task and on the shared table airspace,
              renewed by heartbeats. Before each physical commitment
              (entering the airspace, closing the gripper) it checks the lease
              on its own clock; if it cannot prove it still owns the task, it
              backs off. After the grasp it is committed (physics can't be
              undone) and finishes. It deduplicates commands by order id, and
              after a reboot it must re-validate its job before moving.
"""
from __future__ import annotations

import numpy as np

from .scene import BOX_HALF

# lease timing (seconds)
L_TASK = 2.0
L_AIR = 2.0
LEASE_MARGIN = 0.3   # robot treats its lease as expired this much early
HB_PERIOD = 0.4


class RobotNode:
    def __init__(self, idx, mode, arm, world, net, layout, log):
        self.idx, self.mode, self.arm, self.world, self.net = idx, mode, arm, world, net
        self.name = f"arm{idx}"
        self.lay, self.log = layout, log
        self.staging = np.array(layout.staging[idx])
        bx, by = layout.bin_pos[idx]
        self.bin_above = np.array([bx, by, 0.30])
        self.bin_low = np.array([bx, by, 0.20])
        # --- persisted (survives a crash) ---
        self.queue: list[dict] = []
        self.cmd: dict | None = None
        self.done_orders: set[str] = set()
        self.best_token: dict[str, int] = {}
        # --- volatile ---
        self.phase = "IDLE"
        self.phase_t = 0.0
        self.target_box = None
        self.granted_box = None          # the one pick our airspace reservation covers
        self.air = False                 # baseline: grant received
        self.air_token = None
        self.task_until = -1.0
        self.air_until = -1.0
        self.last_hb = -1.0
        self.last_req = -1.0
        self.pending_done: dict | None = None
        self.crashed = False
        self.status = "idle"
        # ablations of our runtime (see ablation_experiments.py); empty = full runtime
        self.ab: set[str] = set()

    # ------------------------------------------------------------------ utils
    def send(self, t, msg):
        msg["from"] = self.name
        self.net.send(t, self.name, "coord", msg)

    def _go(self, t, phase, target=None, speed=0.75):
        self.phase, self.phase_t = phase, t
        if target is not None:
            self.arm.move_to(target, t, speed=speed)

    def _lease_ok(self, t):
        return self.mode == "baseline" or "no_leases" in self.ab or t < self.task_until

    def _air_ok(self, t):
        if self.mode == "baseline":
            return self.air
        if "no_leases" in self.ab:               # ablation: a grant never expires on the robot
            return self.air_token is not None
        return self.air_token is not None and t < self.air_until

    def committed(self):
        return self.world.holding(self.idx) is not None

    # ------------------------------------------------------------ crash/restart
    def crash(self, t):
        self.crashed = True
        self.arm.frozen = True
        self.net.down.add(self.name)
        self.phase = "CRASHED"
        self.status = "crashed"
        # volatile state is lost
        self.air, self.air_token, self.task_until, self.air_until = False, None, -1.0, -1.0
        self.granted_box = None
        self.log(t, f"{self.name}: CRASH (controller down)")

    def restart(self, t):
        self.crashed = False
        self.arm.frozen = False
        self.net.down.discard(self.name)
        self.arm.hold_here()   # motion plan is volatile: lost in the crash
        self.log(t, f"{self.name}: rebooted")
        if self.cmd is None:
            self._go(t, "RETREAT", self.staging)
            return
        if self.mode == "baseline" or "no_fencing" in self.ab:
            self.log(t, f"{self.name}: resuming persisted job {self.cmd['order']}")
            self._resume(t)
        else:
            self.send(t, {"type": "HELLO", "order": self.cmd["order"], "token": self.cmd["token"],
                          "holding": self.committed()})
            self._go(t, "HELLO_WAIT")
            self.last_req = t

    def d_q(self):
        return self.arm.d.qpos[self.arm.qadr].copy()

    def _resume(self, t):
        if self.committed():
            p = self.arm.tcp_pos()
            # "lift" never means "move down": a rebooted arm may already be above 0.30
            self._go(t, "LIFT", np.array([p[0], p[1], max(p[2], 0.30)]))
        else:
            self._go(t, "GO_STAGE", self.staging)

    # --------------------------------------------------------------- messages
    def on_msg(self, t, msg):
        if self.crashed:
            return
        typ = msg["type"]
        if typ == "ASSIGN":
            o = msg["order"]
            self.send(t, {"type": "ACK", "order": o, "token": msg.get("token")})
            if self.mode == "baseline":
                known = (self.cmd and self.cmd["order"] == o) or any(q["order"] == o for q in self.queue)
                if not known:
                    self.queue.append(dict(msg))
            else:
                best = self.best_token.get(o, -1)
                if "no_fencing" not in self.ab and msg["token"] < best:
                    return                       # fencing: a stale (superseded) command
                if "no_dedup" not in self.ab and (o in self.done_orders or msg["token"] == best):
                    return                       # dedup: this exact command was already taken
                self.best_token[o] = max(best, msg["token"])
                self.queue = [q for q in self.queue if q["order"] != o]
                self.queue.append(dict(msg))
                self.task_until = t + L_TASK - LEASE_MARGIN
        elif typ == "AIR_GRANT":
            if self.mode == "baseline":
                self.air = True
                if self.phase == "WAIT_AIR":
                    self.granted_box = msg.get("box")
            elif self.phase != "WAIT_AIR" and msg["token"] != self.air_token:
                # late/duplicate grant we no longer want: hand it straight back (no lease leaks)
                self.send(t, {"type": "AIR_REL", "token": msg["token"]})
            else:
                self.air_token = msg["token"]
                self.air_until = t + L_AIR - LEASE_MARGIN
                if self.phase == "WAIT_AIR":
                    self.granted_box = msg.get("box")
        elif typ == "RENEW":
            if self.cmd and msg.get("order") == self.cmd["order"] and msg.get("token") == self.cmd["token"]:
                self.task_until = t + L_TASK - LEASE_MARGIN
            if msg.get("air_token") is not None and msg["air_token"] == self.air_token:
                self.air_until = t + L_AIR - LEASE_MARGIN
        elif typ == "REVOKE":
            if "no_fencing" in self.ab:
                return                           # ablation: no tokens, so nothing to revoke against
            if self.cmd and msg["order"] == self.cmd["order"] and msg["token"] >= self.cmd["token"]:
                if not self.committed() and self.phase not in ("ABORT",):
                    self.log(t, f"{self.name}: job {msg['order']} revoked (stale token) -> backing off")
                    self._abort(t)
            self.queue = [q for q in self.queue if not (q["order"] == msg["order"] and q["token"] <= msg["token"])]
        elif typ == "HELLO_REPLY":
            if self.phase != "HELLO_WAIT":
                return
            if msg["valid"]:
                self.cmd["token"] = msg["token"]
                self.best_token[self.cmd["order"]] = msg["token"]
                self.task_until = t + L_TASK - LEASE_MARGIN
                self.log(t, f"{self.name}: job {self.cmd['order']} re-validated (token {msg['token']})")
                self._resume(t)
            else:
                self.log(t, f"{self.name}: job {self.cmd['order']} was reassigned -> dropping it (fenced)")
                self._abort(t)
        elif typ == "DONE_ACK":
            if self.pending_done and self.pending_done["order"] == msg["order"]:
                self.pending_done = None

    # ------------------------------------------------------------------ tick
    def tick(self, t):
        if self.crashed:
            return
        if self.mode == "reliable" and t - self.last_hb >= HB_PERIOD:
            self.last_hb = t
            self.send(t, {"type": "HB", "phase": self.phase,
                          "order": self.cmd["order"] if self.cmd else None,
                          "token": self.cmd["token"] if self.cmd else None,
                          "air_token": self.air_token, "done": self.pending_done})
        ph = self.phase
        a = self.arm
        if ph == "HELLO_WAIT":
            if t - self.last_req > 0.5:
                self.last_req = t
                self.send(t, {"type": "HELLO", "order": self.cmd["order"], "token": self.cmd["token"],
                              "holding": self.committed()})
            self.status = "waiting"
            return
        if ph == "IDLE":
            self.status = "idle"
            if self.queue:
                self.cmd = self.queue.pop(0)
                self.target_box = None
                self.log(t, f"{self.name}: start {self.cmd['order']} ({self.cmd['color']})")
                self._go(t, "GO_STAGE", self.staging)
            return
        self.status = "busy"
        # ---- reliable safety gate before commitment ----
        if self.mode == "reliable" and ph in ("WAIT_AIR", "REACH", "DESCEND", "GRASP") and not self.committed():
            if not self._lease_ok(t):
                self.log(t, f"{self.name}: task lease lost before grasp -> abort safely")
                self._abort(t)
                return
            if ph != "WAIT_AIR" and not self._air_ok(t):
                self.log(t, f"{self.name}: airspace lease lost -> abort safely")
                self._abort(t)
                return
        if ph == "GO_STAGE":
            if a.done(t):
                self._go(t, "WAIT_AIR")
                self.last_req = -1
        elif ph == "WAIT_AIR":
            cands = self._candidates()
            if not cands:
                self.log(t, f"{self.name}: no {self.cmd['color']} box left on table for {self.cmd['order']}")
                self.send(t, {"type": "FAIL", "order": self.cmd["order"], "token": self.cmd.get("token"),
                              "reason": "no_stock"})
                self._release_air(t)
                self.cmd = None
                self._go(t, "IDLE")
                return
            if self._air_ok(t):
                k = self.granted_box
                if k is not None and self.world.box_on_table(k):
                    # the reservation covers exactly this pick: take this box, no other
                    self.target_box = k
                    p = self.world.box_pos(k)
                    self._go(t, "REACH", p + np.array([0, 0, 0.14]))
                    return
                self._release_air(t)          # stale grant (box gone): ask again
            if t - self.last_req > 0.5:
                self.last_req = t
                self.send(t, {"type": "REQ_AIR", "order": self.cmd["order"], "token": self.cmd.get("token"),
                              "cands": cands})
        elif ph == "REACH":
            if a.done(t):
                p = self.world.box_pos(self.target_box)
                self._go(t, "DESCEND", p + np.array([0, 0, 0.0]), speed=0.4)
        elif ph == "DESCEND":
            if a.done(t):
                a.set_grip(True)
                self._go(t, "GRASP")
        elif ph == "GRASP":
            if t - self.phase_t > 0.35:
                if self.world.try_grasp(self.idx, self.target_box, self.cmd["order"]):
                    self.send(t, {"type": "PROGRESS", "order": self.cmd["order"], "phase": "GRASPED"})
                    p = a.tcp_pos()
                    self._go(t, "LIFT", np.array([p[0], p[1], 0.30]), speed=0.55)
                else:
                    a.set_grip(False)
                    self.log(t, f"{self.name}: grasp missed box{self.target_box}, retrying")
                    self._abort_to_wait(t)
        elif ph == "LIFT":
            if a.done(t):
                self._go(t, "EXIT", self.staging)
        elif ph == "EXIT":
            if a.done(t):
                self._release_air(t)
                # pick a free slot in the bin (4 per layer) so boxes don't pile up and tumble out
                n = sum(1 for s in self.world.shipments if s.bin_arm == self.idx)
                dx, dy = [(-0.04, -0.04), (0.04, 0.04), (-0.04, 0.04), (0.04, -0.04)][n % 4]
                bx, by = self.lay.bin_pos[self.idx]
                self.bin_above = np.array([bx + dx, by + dy, 0.30])
                self.bin_low = np.array([bx + dx, by + dy, 0.075 + 0.05 * (n // 4)])
                self._go(t, "TO_BIN", self.bin_above)
        elif ph == "TO_BIN":
            if a.done(t):
                self._go(t, "LOWER", self.bin_low, speed=0.4)
        elif ph == "LOWER":
            if a.done(t):
                a.set_grip(False)
                self._go(t, "OPEN")
        elif ph == "OPEN":
            if t - self.phase_t > 0.3:
                self.world.release(self.idx)
                o = self.cmd["order"]
                self.done_orders.add(o)
                done = {"type": "DONE", "order": o, "token": self.cmd.get("token")}
                self.send(t, done)
                if self.mode == "reliable" and "no_done_retry" not in self.ab:
                    self.pending_done = done     # repeat DONE in heartbeats until acknowledged
                self.log(t, f"{self.name}: shipped {o}, reported DONE")
                self.cmd = None
                self._go(t, "RETREAT", self.staging)
        elif ph == "RETREAT":
            if a.done(t):
                self._go(t, "IDLE")
        elif ph == "REGRASP":
            if a.done(t):
                if np.linalg.norm(a.tcp_pos() - self.staging) > 0.03:
                    a.move_to(self.staging, t)
                else:
                    self._release_air(t)
                    self._go(t, "WAIT_AIR")
                    self.last_req = -1
        elif ph == "ABORT":
            if a.done(t):
                if np.linalg.norm(a.tcp_pos() - self.staging) > 0.03:
                    a.move_to(self.staging, t)
                else:
                    self._release_air(t)
                    self._go(t, "IDLE")
        if self.phase == "ABORT":
            self.status = "aborting"

    def _candidates(self):
        """Perception: boxes of the right colour on the table, hinted box first, then nearest."""
        me = self.arm.tcp_pos()
        cands = sorted(self.world.boxes_on_table(self.cmd["color"]),
                       key=lambda k: np.linalg.norm(self.world.box_pos(k) - me))
        hint = self.cmd.get("box")
        if hint in cands:
            cands.remove(hint)
            cands.insert(0, hint)
        return [int(k) for k in cands]

    def _abort_to_wait(self, t):
        """Missed grasp: climb back out of the airspace, then queue for a new reservation."""
        p = self.arm.tcp_pos()
        self._go(t, "REGRASP", np.array([p[0], p[1], 0.30]))

    def _release_air(self, t):
        if self.air or self.air_token is not None:
            self.send(t, {"type": "AIR_REL", "token": self.air_token})
        self.air, self.air_token, self.air_until = False, None, -1.0
        self.granted_box = None

    def _abort(self, t):
        self.arm.set_grip(False)
        self.cmd = None
        self.target_box = None
        p = self.arm.tcp_pos()
        # go straight up first so we never sweep through the table
        up = np.array([p[0], p[1], max(p[2], 0.30)])
        self._go(t, "ABORT", up if p[2] < 0.28 else self.staging)

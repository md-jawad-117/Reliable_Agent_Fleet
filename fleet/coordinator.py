"""The coordinator: the planning agent that turns customer orders into robot jobs.

The *planning policy* (which arm, which box) is identical in both systems and is
deliberately simple; it is the slot where an LLM agent plugs in. What differs is
the *execution protocol* around it:

  BaselineCoordinator  - at-least-once delivery with retries, a timeout-based
                         airspace lock, and timeout-based job reassignment.
                         This is how most agent frameworks treat tool calls.

  ReliableCoordinator  - an order ledger with fencing tokens and leases,
                         heartbeat-based failure detection, and *physical
                         reconciliation*: before it retries or reassigns anything
                         uncertain it checks the cell camera (ground truth),
                         because a physical action cannot be rolled back.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .robot import L_AIR, L_TASK
from .zones import compatible

ORDER_COLORS = ["red", "green", "blue", "green", "blue", "red", "blue", "red", "green"]
SKEW = 0.3                   # coordinator waits this long beyond a lease before acting
BASE_TASK_TIMEOUT = 15.0     # baseline: reassign a job not reported DONE in this time
BASE_AIR_TIMEOUT = 5.0       # baseline: force-release the airspace lock after this
RESEND = 1.0
RESEND_FAST = 0.3            # reliable: resend un-acked assignments quickly (they're idempotent)


@dataclass
class Order:
    oid: str
    color: str
    state: str = "PENDING"        # PENDING | ASSIGNED | DONE | FAILED
    arm: int | None = None
    box: int | None = None
    token: int = -1
    t_assign: float = 0.0
    acked: bool = False
    last_send: float = -1.0
    lease_until: float = 0.0
    excluded: set = field(default_factory=set)
    note: str = ""


class _Base:
    def __init__(self, net, world, n_arms, log):
        self.net, self.world, self.log = net, world, log
        self.n_arms = n_arms
        self.orders = [Order(f"O{i + 1}", c) for i, c in enumerate(ORDER_COLORS)]
        self.by_id = {o.oid: o for o in self.orders}
        self.box_free = {k: True for k in range(world.n_box)}     # coordinator's *belief*
        self.arm_busy = [None] * n_arms                           # order id believed in progress
        # fine-grained airspace: each holder reserves one pick (arm, box); holders must be
        # pairwise compatible according to the precomputed path-conflict table (zones.py)
        self.holders: dict[int, dict] = {}       # arm -> {token, box, since, until}
        self.air_waiting: dict[int, dict] = {}   # arm -> latest REQ_AIR (FIFO by first request)
        self.air_token = 0
        self.token = 0
        self.planner = None          # optional LLM planner (llm_planner.LLMPlanner)
        self.ab: set[str] = set()    # ablations of our runtime; empty = full runtime

    def arm_ready(self, t, arm) -> bool:
        """Would this coordinator accept a new order for `arm` right now?"""
        return self.arm_busy[arm] is None

    def _note(self, t, text):
        """Tell the LLM planner (if any) that something happened."""
        if self.planner is not None:
            self.planner.note(t, text)

    def send(self, t, arm, msg):
        self.net.send(t, "coord", f"arm{arm}", msg)

    # ---- airspace scheduling (shared by both systems; only failure handling differs)
    def _physically_inside(self, arm) -> bool:
        return False

    def _on_req_air(self, t, arm, msg):
        msg = dict(msg, t=t)
        h = self.holders.get(arm)
        if h is not None:
            if h["box"] in msg.get("cands", []):
                # re-sending a grant re-starts the robot's lease clock, so it must re-start ours
                # too; otherwise we could free a reservation the robot still believes it holds
                h["until"] = max(h["until"], t + L_AIR)
                self.send(t, arm, {"type": "AIR_GRANT", "token": h["token"], "box": h["box"]})
                return
            if not self._physically_inside(arm):
                self.holders.pop(arm)       # its old reservation is useless to it now
        first = self.air_waiting.get(arm, {}).get("first", t)
        self.air_waiting[arm] = dict(msg, first=first)

    def _order_ok(self, o, arm, req) -> bool:
        return o is not None and o.state == "ASSIGNED" and o.arm == arm

    def _pick_compatible(self, arm, req):
        oid = req.get("order")
        held = {h["box"] for h in self.holders.values()}
        for k in req.get("cands", []):
            if self.box_free.get(k) not in (True, oid) or k in held:
                continue
            if all(compatible(arm, k, b, h["box"]) for b, h in self.holders.items() if b != arm):
                return k
        return None

    def _grant_waiting(self, t, may_grant=lambda arm: True, until=0.0):
        for arm in sorted(self.air_waiting, key=lambda a: self.air_waiting[a]["first"]):
            req = self.air_waiting[arm]
            if t - req["t"] > 1.5:              # robot stopped asking
                self.air_waiting.pop(arm)
                continue
            if arm in self.holders or not may_grant(arm):
                continue
            k = self._pick_compatible(arm, req)
            if k is None:
                continue
            self.air_token += 1
            self.holders[arm] = {"token": self.air_token, "box": k, "since": t, "until": t + until}
            self.air_waiting.pop(arm)
            o = self.by_id.get(req.get("order"))
            if self._order_ok(o, arm, req) and o.box != k:      # planner swaps to a compatible box
                if o.box is not None and self.box_free.get(o.box) == o.oid:
                    self.box_free[o.box] = True
                o.box = k
                self.box_free[k] = o.oid
            self.send(t, arm, {"type": "AIR_GRANT", "token": self.air_token, "box": k})
            if len(self.holders) > 1:
                self.log(t, f"coord: arm{arm} -> box{k} in parallel with "
                            + ", ".join(f"arm{b}" for b in self.holders if b != arm))

    def all_settled(self):
        return all(o.state in ("DONE", "FAILED") for o in self.orders)

    def _choose_box(self, o: Order):
        # box_free[k] is True (free), an order id (reserved by that order) or False (consumed)
        if o.box is not None and self.box_free.get(o.box) in (True, o.oid):
            return o.box
        for k, free in self.box_free.items():
            if free is True and self.world.box_color[k] == o.color:
                return k
        return None

    def _assign(self, t, o: Order, arm: int):
        box = self._choose_box(o)
        if box is None:
            o.state = "FAILED"
            o.note = "out of stock"
            self.log(t, f"coord: {o.oid} FAILED - no {o.color} box left")
            self._note(t, f"{o.oid} FAILED: no {o.color} box left")
            return
        self.box_free[box] = o.oid
        self.token += 1
        o.state, o.arm, o.box, o.token, o.t_assign, o.acked = "ASSIGNED", arm, box, self.token, t, False
        o.lease_until = t + L_TASK
        o.last_send = t
        self.arm_busy[arm] = o.oid
        self.send(t, arm, self._assign_msg(o))
        self.log(t, f"coord: {o.oid} ({o.color}) -> arm{arm}, box{box}, token {o.token}")

    def _assign_msg(self, o):
        return {"type": "ASSIGN", "order": o.oid, "color": o.color, "box": o.box, "token": o.token}


# ============================================================================ baseline
class BaselineCoordinator(_Base):
    name = "baseline"

    def on_msg(self, t, src, msg):
        arm = int(src[3])
        typ = msg["type"]
        o = self.by_id.get(msg.get("order"))
        if typ == "ACK" and o and o.arm == arm:
            o.acked = True
        elif typ == "DONE" and o:
            if o.state == "DONE":
                self.log(t, f"coord: DONE for {o.oid} again (already done)")
            else:
                self._note(t, f"arm{arm} reported {o.oid} DONE")
            o.state = "DONE"
            if o.box is not None:
                self.box_free[o.box] = False
            if self.arm_busy[arm] == o.oid:
                self.arm_busy[arm] = None
        elif typ == "FAIL" and o:
            self._note(t, f"arm{arm} reported {o.oid} FAILED ({msg.get('reason')})")
            if o.box is not None:
                self.box_free[o.box] = False
            o.box = None
            if o.state == "ASSIGNED" and o.arm == arm:
                o.state = "PENDING"
            if self.arm_busy[arm] == o.oid:
                self.arm_busy[arm] = None
        elif typ == "REQ_AIR":
            self._on_req_air(t, arm, msg)
        elif typ == "AIR_REL":
            self.holders.pop(arm, None)

    def tick(self, t):
        # airspace reservations with timeout
        for arm, h in list(self.holders.items()):
            if t - h["since"] > BASE_AIR_TIMEOUT:
                self.log(t, f"coord: airspace reservation timed out for arm{arm} -> force release")
                self.holders.pop(arm)
        self._grant_waiting(t)
        for o in self.orders:
            if o.state != "ASSIGNED":
                continue
            if not o.acked and t - o.last_send > RESEND:
                o.last_send = t
                self.send(t, o.arm, self._assign_msg(o))
            if self.planner is None and t - o.t_assign > BASE_TASK_TIMEOUT:
                self.log(t, f"coord: {o.oid} timed out on arm{o.arm} -> reassigning")
                o.excluded.add(o.arm)
                if self.arm_busy[o.arm] == o.oid:
                    self.arm_busy[o.arm] = None
                o.state = "PENDING"
        if self.planner is not None:
            self.planner.maybe_decide(t, self)
            return
        for o in self.orders:
            if o.state != "PENDING":
                continue
            free = [a for a in range(self.n_arms) if self.arm_busy[a] is None and a not in o.excluded]
            if not free:
                free = [a for a in range(self.n_arms) if self.arm_busy[a] is None]
            if free:
                self._assign(t, o, free[0])

    def llm_assign(self, t, o, arm):
        """Typical agent stack: the tool call goes straight to the robot, no questions asked."""
        if o.state == "ASSIGNED" and o.arm is not None and o.arm != arm and self.arm_busy[o.arm] == o.oid:
            self.arm_busy[o.arm] = None
        if o.state in ("DONE", "FAILED"):
            o.state = "PENDING"
        self._assign(t, o, arm)
        if o.state != "ASSIGNED":
            return False, o.note or "could not assign"
        return True, f"command sent to arm{arm}"


# ============================================================================ reliable
class ReliableCoordinator(_Base):
    name = "reliable"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.last_seen = [-10.0] * self.n_arms
        self.arm_phase = ["IDLE"] * self.n_arms
        self.air_until = 0.0
        self.reconcile_note: dict[str, str] = {}

    def alive(self, t, arm):
        return t - self.last_seen[arm] < 1.2

    def _physically_inside(self, arm):
        return "no_camera" not in self.ab and arm in self.world.arms_in_airspace()

    def arm_ready(self, t, arm):
        return self.arm_busy[arm] is None and self.alive(t, arm) and self.arm_phase[arm] == "IDLE"

    def _order_ok(self, o, arm, req):
        return super()._order_ok(o, arm, req) and req.get("token") == o.token

    # ---- messages ----
    def on_msg(self, t, src, msg):
        arm = int(src[3])
        typ = msg["type"]
        self.last_seen[arm] = t
        o = self.by_id.get(msg.get("order"))
        if typ == "ACK" and o and o.arm == arm and msg.get("token") == o.token:
            o.acked = True
        elif typ == "HB":
            self.arm_phase[arm] = msg["phase"]
            if msg.get("done"):
                self._on_done(t, arm, msg["done"])
            reply = {"type": "RENEW", "order": None, "token": None, "air_token": None}
            if o is not None:
                if o.state == "ASSIGNED" and o.arm == arm and o.token == msg["token"]:
                    o.lease_until = t + L_TASK
                    o.acked = True
                    reply["order"], reply["token"] = o.oid, o.token
                elif not (o.state == "DONE" and o.arm == arm):
                    self.send(t, arm, {"type": "REVOKE", "order": o.oid, "token": msg["token"]})
            h = self.holders.get(arm)
            if h is not None and msg.get("air_token") == h["token"]:
                h["until"] = t + L_AIR
                reply["air_token"] = h["token"]
            self.send(t, arm, reply)
        elif typ == "DONE":
            self._on_done(t, arm, msg)
        elif typ == "FAIL" and o and o.arm == arm and msg.get("token") == o.token:
            self._note(t, f"arm{arm} reported {o.oid} FAILED ({msg.get('reason')})")
            if o.box is not None:
                self.box_free[o.box] = False
            o.box = None
            o.state = "PENDING"
            if self.arm_busy[arm] == o.oid:
                self.arm_busy[arm] = None
        elif typ == "REQ_AIR":
            self._on_req_air(t, arm, msg)
        elif typ == "AIR_REL":
            h = self.holders.get(arm)
            if h is not None and msg.get("token") == h["token"]:
                self.holders.pop(arm)
        elif typ == "HELLO":
            if o and o.state == "ASSIGNED" and o.arm == arm:
                self.token += 1
                o.token = self.token
                o.lease_until = t + L_TASK
                self.send(t, arm, {"type": "HELLO_REPLY", "valid": True, "token": o.token})
                self.log(t, f"coord: arm{arm} back; {o.oid} still its job -> new token {o.token}")
            else:
                self.send(t, arm, {"type": "HELLO_REPLY", "valid": False})
                self.log(t, f"coord: arm{arm} back with stale job {msg.get('order')} -> fenced off")

    def _on_done(self, t, arm, msg):
        o = self.by_id.get(msg["order"])
        self.send(t, arm, {"type": "DONE_ACK", "order": msg["order"]})
        if o is None or o.state == "DONE":
            return
        if o.arm == arm and o.token == msg.get("token"):
            self._mark_done(t, o, "robot report")

    def _mark_done(self, t, o, how):
        o.state = "DONE"
        if o.box is not None:
            self.box_free[o.box] = False
        if o.arm is not None and self.arm_busy[o.arm] == o.oid:
            self.arm_busy[o.arm] = None
        if how != "robot report":
            self.log(t, f"coord: {o.oid} confirmed DONE via {how}")
        self._note(t, f"{o.oid} DONE (confirmed by {how})")

    # ---- periodic ----
    def tick(self, t):
        # --- airspace leases: an expired reservation is freed only when the camera shows
        #     that arm physically out of the airspace; until then it still blocks conflicting picks
        no_leases, no_camera = "no_leases" in self.ab, "no_camera" in self.ab
        inside = [] if no_camera else self.world.arms_in_airspace()
        for arm, h in list(self.holders.items()):
            if no_leases:      # ablation: a fixed timeout instead of a heartbeat-renewed lease
                if t - h["since"] <= BASE_AIR_TIMEOUT:
                    continue
            elif t <= h["until"] + SKEW:
                continue
            key = f"air{arm}"
            if arm in inside:
                if key not in self.reconcile_note:
                    self.reconcile_note[key] = "1"
                    self.log(t, f"coord: airspace lease of arm{arm} expired but camera sees it inside "
                                f"-> keep its reservation (box{h['box']})")
            else:
                self.log(t, f"coord: airspace lease of arm{arm} expired, camera: out -> free its reservation")
                self.holders.pop(arm)
                self.reconcile_note.pop(key, None)

        # grant only to live arms, and only if every arm the camera sees in the airspace
        # is accounted for by a reservation (an unknown arm in there could be anywhere)
        def may_grant(arm):
            return self.alive(t, arm) and all(a == arm or a in self.holders for a in inside)

        self._grant_waiting(t, may_grant, until=L_AIR)
        # --- job leases
        for o in self.orders:
            if o.state != "ASSIGNED":
                continue
            if not o.acked and t - o.last_send > RESEND_FAST:
                o.last_send = t
                self.send(t, o.arm, self._assign_msg(o))
            expired = (t - o.t_assign > BASE_TASK_TIMEOUT) if no_leases else (t > o.lease_until + SKEW)
            if expired:
                self._reconcile(t, o)
        # --- new assignments only to arms that are alive and idle
        if self.planner is not None:
            self.planner.maybe_decide(t, self)
            return
        for o in self.orders:
            if o.state != "PENDING":
                continue
            free = [a for a in range(self.n_arms)
                    if self.arm_busy[a] is None and self.alive(t, a) and self.arm_phase[a] == "IDLE"
                    and a not in o.excluded]
            if not free:
                free = [a for a in range(self.n_arms)
                        if self.arm_busy[a] is None and self.alive(t, a) and self.arm_phase[a] == "IDLE"]
            if free:
                self._assign(t, o, free[0])

    def llm_assign(self, t, o, arm):
        """Our runtime: the agent's tool call is an intent, checked before anything moves."""
        if o.state == "DONE":
            return False, f"{o.oid} is already shipped (confirmed). Nothing to do."
        if o.state == "FAILED":
            return False, f"{o.oid} has failed: {o.note}."
        if o.state == "ASSIGNED":
            if o.arm == arm:
                return False, (f"{o.oid} is already assigned to arm{arm}; the runtime keeps re-delivering "
                               f"the command until the robot acknowledges it.")
            return False, (f"{o.oid} is in progress on arm{o.arm} under a lease. If arm{o.arm} fails, the "
                           f"runtime fences it, checks the cell camera, and returns {o.oid} to PENDING "
                           f"only if its box was not picked. Re-assigning now could ship it twice.")
        if not self.alive(t, arm):
            return False, f"arm{arm} is unreachable (no heartbeat). Choose another arm or wait."
        if self.arm_busy[arm] is not None or self.arm_phase[arm] != "IDLE":
            return False, f"arm{arm} is busy. Wait until it is idle."
        self._assign(t, o, arm)
        if o.state != "ASSIGNED":
            return False, o.note or "could not assign"
        return True, f"assigned {o.oid} to arm{arm} (token {o.token})"

    def _reconcile(self, t, o: Order):
        """A job's lease lapsed: we do not know what the robot did. Ask the physical world."""
        if "no_camera" in self.ab:     # ablation: no ground truth, so assume "not done" and redo
            status, k = "none", None
        else:
            status, k = self.world.camera_box_status(o.arm, o.t_assign)
        if status == "shipped":
            self._mark_done(t, o, f"camera (box{k} is in arm{o.arm}'s bin)")
            return
        if status in ("held", "moving"):
            key = f"{o.oid}:held"
            if key not in self.reconcile_note:
                self.reconcile_note[key] = "1"
                self.log(t, f"coord: lost contact with arm{o.arm}, camera shows box{k} already picked for "
                            f"{o.oid} -> committed, wait (do NOT reassign)")
            return
        if status == "dropped":
            self.box_free[k] = False
            self.log(t, f"coord: camera shows box{k} dropped outside the bin -> written off")
        self.log(t, f"coord: {o.oid} lease expired on arm{o.arm}, camera: not picked -> fence token "
                    f"{o.token} and reassign")
        self._note(t, f"runtime: lost contact with arm{o.arm}; camera shows {o.oid}'s box not picked; "
                      f"arm{o.arm} fenced off, {o.oid} is PENDING again")
        self.send(t, o.arm, {"type": "REVOKE", "order": o.oid, "token": o.token})
        o.excluded.add(o.arm)
        if self.arm_busy[o.arm] == o.oid:
            self.arm_busy[o.arm] = None
        o.state = "PENDING"
        if o.box is not None and not self.world.box_on_table(o.box):
            self.box_free[o.box] = False
            o.box = None

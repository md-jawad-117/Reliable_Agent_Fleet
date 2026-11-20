"""An LLM agent as the fleet's planner.

The scripted assignment loop in coordinator.py is replaced by Claude making tool
calls. The agent only *proposes*: `assign(order, arm)`. What happens to a proposal
depends on the runtime underneath it:

  baseline  - like most agent frameworks: the tool call goes straight to the robot.
              If the agent re-assigns an order that is still in progress (because a
              "done" was lost, or the arm looks slow), a second arm runs it.

  reliable  - our runtime treats the call as an *intent* and checks it against the
              order ledger, leases and the arm's liveness first. Unsafe intents are
              refused with a reason the agent can read; recovery (fencing + camera
              reconciliation) stays in the runtime, which hands an order back to the
              agent as PENDING only once it is provably safe to redo.

Both systems show the agent exactly the same state view, so any difference in what
reaches the physical world comes from the runtime, not from better information.

The simulation clock is frozen while the agent thinks (decision latency is not
modelled); each decision is a fresh, bounded conversation built from current state.
"""
from __future__ import annotations

import json
import time
from collections import Counter

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM = """You are the planning agent for a robot work cell. Three robot arms (arm0, arm1, arm2) \
pick coloured boxes from a shared table and drop them in their own shipping bin. Each customer \
order needs exactly one box of its colour shipped, exactly once.

Every time you are called you get the current state. Decide what to do with the `assign` tool: \
give PENDING orders to idle arms (one order per arm at a time). You may also re-assign an order \
that you believe is stuck. Call `assign` once per order you want to (re)assign; several calls in \
one turn are fine. If nothing needs doing, reply with a short sentence and no tool call.

A shipped box cannot be un-shipped: avoid having two arms ship the same order."""

TOOLS = [{
    "name": "assign",
    "description": "Assign (or re-assign) a customer order to a robot arm. The arm picks a box of "
                   "the order's colour and ships it. Returns whether the command was accepted.",
    "input_schema": {
        "type": "object",
        "properties": {
            "order": {"type": "string", "description": "Order id, e.g. 'O3'."},
            "arm": {"type": "integer", "enum": [0, 1, 2], "description": "Arm index."},
            "reason": {"type": "string", "description": "One short sentence: why."},
        },
        "required": ["order", "arm", "reason"],
        "additionalProperties": False,
    },
    "strict": True,
}]


class LLMPlanner:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low", client=None,
                 stuck_after: float = 12.0, recheck_every: float = 6.0, max_calls: int = 80):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.effort = client, model, effort
        self.stuck_after, self.recheck_every, self.max_calls = stuck_after, recheck_every, max_calls
        self.last_digest = None
        self.last_call_t = -1e9
        self._refused_normal_last = False
        self.events: list[str] = []          # what the agent has been told happened
        self.stats = Counter()
        self.decisions: list[dict] = []      # audit log of every tool call

    # ------------------------------------------------------------------ trigger
    def _digest(self, t, coord):
        return (tuple(o.state for o in coord.orders), tuple(coord.arm_ready(t, a) for a in range(coord.n_arms)))

    def maybe_decide(self, t, coord):
        if self.stats["calls"] >= self.max_calls:
            return
        pending = [o for o in coord.orders if o.state == "PENDING"]
        idle = [a for a in range(coord.n_arms) if coord.arm_ready(t, a)]
        stuck = [o for o in coord.orders if o.state == "ASSIGNED" and t - o.t_assign > self.stuck_after]
        digest = self._digest(t, coord)
        changed = digest != self.last_digest
        # after a refused normal intent (arm still busy / not yet reachable) look again soon
        wait = 1.0 if self._refused_normal_last else self.recheck_every
        quiet = t - self.last_call_t > wait
        if (pending and idle and (changed or quiet)) or (stuck and quiet):
            self.last_digest = digest
            self.last_call_t = t
            self.decide(t, coord)

    # ------------------------------------------------------------------ prompt
    def state_text(self, t, coord) -> str:
        lines = [f"Time: t = {t:.1f} s", "", "Orders:"]
        for o in coord.orders:
            if o.state == "ASSIGNED":
                s = (f"ASSIGNED to arm{o.arm} {t - o.t_assign:.1f} s ago, "
                     f"robot acknowledged: {'yes' if o.acked else 'no'}")
            elif o.state in ("DONE", "FAILED"):
                s = o.state + (f" ({o.note})" if o.note else "")
            else:
                s = "PENDING"
            lines.append(f"  {o.oid} {o.color}: {s}")
        lines += ["", "Arms:"]
        for a in range(coord.n_arms):
            b = coord.arm_busy[a]
            lines.append(f"  arm{a}: " + (f"busy with {b}" if b else
                                          "idle" if coord.arm_ready(t, a) else "not ready"))
        stock = Counter(coord.world.box_color[k] for k in coord.world.boxes_on_table())
        lines += ["", "Boxes on the table: " + ", ".join(f"{c} {stock.get(c, 0)}" for c in ("red", "green", "blue"))]
        if self.events:
            lines += ["", "Recent events:"] + [f"  {e}" for e in self.events[-12:]]
        return "\n".join(lines)

    def note(self, t, text):
        self.events.append(f"t={t:.1f}s {text}")

    # ------------------------------------------------------------------ one decision
    def decide(self, t, coord):
        before = self.stats["refused_normal"]
        try:
            self._decide(t, coord)
        finally:
            self._refused_normal_last = self.stats["refused_normal"] > before

    def _decide(self, t, coord):
        messages = [{"role": "user", "content": self.state_text(t, coord)}]
        for _ in range(4):                       # bounded tool loop per decision
            t0 = time.time()
            resp = self._create(messages)
            self.stats["calls"] += 1
            self.stats["api_seconds"] += time.time() - t0
            u = getattr(resp, "usage", None)
            if u is not None:
                self.stats["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                self.stats["output_tokens"] += getattr(u, "output_tokens", 0) or 0
            if resp.stop_reason == "refusal":
                self.stats["refusals"] += 1
                return
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                return
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in uses:
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": self._run_tool(t, coord, b.name, b.input)})
            messages.append({"role": "user", "content": results})

    def _create(self, messages):
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def _run_tool(self, t, coord, name, inp) -> str:
        return self.run_tool(t, coord, name, inp)

    def run_tool(self, t, coord, name, inp) -> str:
        if name != "assign":
            return json.dumps({"ok": False, "error": f"unknown tool {name}"})
        oid, arm = str(inp.get("order")), inp.get("arm")
        o = coord.by_id.get(oid)
        if o is None or arm not in range(coord.n_arms):
            self.stats["intents"] += 1
            self.stats["intent_invalid"] += 1
            self.stats["refused"] += 1
            self.decisions.append(dict(t=round(t, 2), order=oid, arm=arm, kind="invalid", ok=False,
                                       msg="unknown order or arm", reason=inp.get("reason", "")))
            return json.dumps({"ok": False, "error": "unknown order or arm"})
        # classify the intent against ground truth (for the paper, not shown to the agent)
        if o.state == "DONE" or (o.state == "ASSIGNED" and o.arm != arm):
            kind = "unsafe_redo"         # could make a second arm ship the same order
        elif o.state == "ASSIGNED" and o.arm == arm:
            kind = "repeat"              # same arm again: a duplicate command
        elif o.state == "PENDING" and not coord.arm_ready(t, arm):
            kind = "wrong_arm"           # busy or unreachable arm
        else:
            kind = "normal"
        self.stats["intents"] += 1
        self.stats[f"intent_{kind}"] += 1
        ok, msg = coord.llm_assign(t, o, arm)
        self.stats["accepted" if ok else "refused"] += 1
        if not ok:
            self.stats[f"refused_{kind}"] += 1
        self.decisions.append(dict(t=round(t, 2), order=oid, arm=arm, kind=kind, ok=ok, msg=msg,
                                   reason=inp.get("reason", "")))
        coord.log(t, f"agent: assign {oid} -> arm{arm} ({kind}) -> {'OK' if ok else 'REFUSED: ' + msg}")
        return json.dumps({"ok": ok, "message": msg})

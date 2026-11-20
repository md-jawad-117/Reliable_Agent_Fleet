"""Agent fault injection: offline stand-ins for an LLM planner that misbehaves on purpose.

`FaultyAgentClient` mimics the shape of `client.beta.messages.create(...)` responses, so it
plugs into `LLMPlanner` exactly where Claude would. It reads the same state text Claude
would see, plays a competent base policy (give each PENDING order to an idle arm), and then
injects one or more *agent faults* - each modelled on a documented way LLM agents go wrong:

  impatient     - an order with no DONE after a while "looks stuck" -> re-assign it elsewhere
                  (retry/timeout logic living inside the agent)
  forgetful     - loses track of what it already did and re-issues an earlier assignment,
                  to the same arm or a different one (context loss, duplicated tool calls)
  confused      - sends an order to the wrong arm: one that is busy, or crashed
                  (mis-reading state)
  hallucinating - "re-does" an order that is already shipped, or invents an order id
                  (acting on beliefs that are not true)
  mixed         - all of the above

`rate` is the probability of each fault per opportunity. Everything is seeded, so a run is
reproducible, and the same agent behaviour is replayed against both runtimes.
"""
from __future__ import annotations

import itertools
import re
from types import SimpleNamespace

import numpy as np

KINDS = ("impatient", "forgetful", "confused", "hallucinating")
_ids = itertools.count()
_NO_USAGE = SimpleNamespace(input_tokens=0, output_tokens=0)


def _text(msg):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=msg)],
                           usage=_NO_USAGE)


class _Messages:
    def __init__(self, faults, rate, seed, patience):
        self.faults, self.rate, self.patience = set(faults), rate, patience
        self.rng = np.random.default_rng(seed)
        self.history: list[tuple[str, int]] = []      # every (order, arm) this agent has issued

    def _p(self, kind):
        return kind in self.faults and self.rng.random() < self.rate

    def create(self, *, messages, **_):
        last = messages[-1]["content"]
        if isinstance(last, list):        # tool results came back: this decision is over
            return _text("ok")
        pending = re.findall(r"^\s+(O\d+) \w+: PENDING", last, re.M)
        done = re.findall(r"^\s+(O\d+) \w+: DONE", last, re.M)
        assigned = [(o, int(a), float(s)) for o, a, s in
                    re.findall(r"^\s+(O\d+) \w+: ASSIGNED to arm(\d) ([\d.]+) s ago", last, re.M)]
        idle = [int(a) for a in re.findall(r"^\s+arm(\d): idle", last, re.M)]
        calls = []

        # competent base policy, possibly confused about which arm to use
        for o in pending:
            if not idle:
                break
            arm = idle.pop(0)
            if self._p("confused"):
                arm = int(self.rng.integers(3))
                calls.append((o, arm, "confused: wrong arm"))
            else:
                calls.append((o, arm, "pending order, idle arm"))
        # impatient: re-assign orders that look stuck
        for o, a, age in assigned:
            if age > self.patience and self._p("impatient"):
                others = [b for b in (0, 1, 2) if b != a]
                calls.append((o, int(self.rng.choice(others)), f"no DONE after {age:.0f}s, assume stuck"))
        # forgetful: re-issue something it already did
        if self.history and self._p("forgetful"):
            o, a = self.history[int(self.rng.integers(len(self.history)))]
            if self.rng.random() < 0.5:
                a = int(self.rng.integers(3))
            calls.append((o, a, "forgetful: re-issued an earlier assignment"))
        # hallucinating: act on something that is not true
        if self._p("hallucinating"):
            if done and self.rng.random() < 0.8:
                calls.append((str(self.rng.choice(done)), int(self.rng.integers(3)),
                              "hallucinating: believes a shipped order is still open"))
            else:
                calls.append((f"O{int(self.rng.integers(10, 20))}", int(self.rng.integers(3)),
                              "hallucinating: invented order"))

        if not calls:
            return _text("nothing to do")
        self.history += [(o, a) for o, a, _ in calls if re.fullmatch(r"O[1-9]", o)]
        content = [SimpleNamespace(type="tool_use", id=f"toolu_{next(_ids)}", name="assign",
                                   input={"order": o, "arm": a, "reason": r}) for o, a, r in calls]
        return SimpleNamespace(stop_reason="tool_use", content=content, usage=_NO_USAGE)


class FaultyAgentClient:
    def __init__(self, faults=("impatient",), rate: float = 1.0, seed: int = 0, patience: float = 8.0):
        if isinstance(faults, str):
            faults = KINDS if faults == "mixed" else (faults,)
        self.beta = SimpleNamespace(messages=_Messages(faults, rate, seed, patience))


def ImpatientAgentClient(patience: float = 14.0, seed: int = 0):
    """Always re-assigns an order once it has gone `patience` seconds without a DONE."""
    return FaultyAgentClient("impatient", rate=1.0, seed=seed, patience=patience)

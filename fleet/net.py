"""Simulated, unreliable message network + deterministic fault injection.

Nodes: "coord", "arm0", "arm1", "arm2".
Messages are delivered after a random latency. Links can drop messages
(one direction or both) and nodes can crash (stop processing) and restart.
All randomness comes from one seeded RNG, so runs are reproducible.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np


class Network:
    def __init__(self, rng: np.random.Generator, lat=(0.02, 0.08), loss=0.0):
        self.rng, self.lat, self.loss = rng, lat, loss
        self._q: list = []
        self._seq = itertools.count()
        self.blocked: set[tuple[str, str]] = set()   # (src, dst) pairs currently dropping
        self.down: set[str] = set()                  # crashed nodes (drop all traffic to them)
        self.sent = 0
        self.dropped = 0

    def send(self, t: float, src: str, dst: str, msg: dict):
        self.sent += 1
        if (src, dst) in self.blocked or src in self.down or self.rng.random() < self.loss:
            self.dropped += 1
            return
        delay = self.rng.uniform(*self.lat)
        heapq.heappush(self._q, (t + delay, next(self._seq), dst, src, msg))

    def deliver(self, t: float):
        out = []
        while self._q and self._q[0][0] <= t:
            _, _, dst, src, msg = heapq.heappop(self._q)
            if dst in self.down or (src, dst) in self.blocked:
                self.dropped += 1
                continue
            out.append((dst, src, msg))
        return out


@dataclass
class Fault:
    """A fault that fires the first time `arm` reaches `phase` at or after `after` seconds."""
    kind: str            # "crash" | "partition" | "mute" (arm->coord messages lost)
    arm: int
    after: float
    duration: float
    phase: str | None = None
    fired_at: float | None = None
    healed: bool = False
    label: str = field(default="")


def demo_faults() -> list[Fault]:
    """The fixed fault script used in the demo video. Same script for both systems."""
    return [
        Fault("mute", arm=1, after=3.0, duration=10.0, phase="TO_BIN",
              label="arm1 -> coordinator messages lost (acks/done never arrive)"),
        Fault("crash", arm=2, after=12.0, duration=9.0, phase="DESCEND",
              label="arm2 controller crashes while reaching into the table"),
        Fault("partition", arm=0, after=24.0, duration=9.0, phase="REACH",
              label="arm0 network partition (both directions)"),
    ]


def random_faults(rng: np.random.Generator, n: int = 3) -> list[Fault]:
    kinds = ["mute", "crash", "partition"]
    phases = ["REACH", "DESCEND", "LIFT", "TO_BIN", None]
    out = []
    for _ in range(n):
        out.append(Fault(kind=str(rng.choice(kinds)), arm=int(rng.integers(3)),
                         after=float(rng.uniform(2, 35)), duration=float(rng.uniform(3, 12)),
                         phase=phases[int(rng.integers(len(phases)))]))
    return out

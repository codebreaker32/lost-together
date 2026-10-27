"""A hiker's (or the group's) map: what has been seen, known exits, and predator evidence that fades.

Solo hikers each own a Knowledge. Under SHARE/GROUP one Knowledge is shared (global comms),
or hikers merge their maps when within comm range. Reports from other hikers carry the
sender's localisation drift, which is the price of merging maps built from imperfect odometry.
"""
from dataclasses import dataclass, field

import numpy as np

from .config import DecisionConfig
from .perception import Observation, Sighting


@dataclass
class Knowledge:
    size: int
    dcfg: DecisionConfig
    seen: np.ndarray = None                      # bool: cell observed
    passable: np.ndarray = None                  # bool: observed and passable
    exits: set = field(default_factory=set)
    sightings: dict[int, Sighting] = field(default_factory=dict)
    version: int = 0                             # bumps when exits or sightings change (triggers replans)

    def __post_init__(self):
        self.seen = np.zeros((self.size, self.size), bool)
        self.passable = np.zeros((self.size, self.size), bool)

    def integrate(self, obs: Observation, offset: np.ndarray | None = None) -> None:
        rr, cc = obs.cells
        self.seen[rr, cc] = True
        self.passable[rr, cc] = obs.passable
        for e in obs.exits:
            if e not in self.exits:
                self.exits.add(e)
                self.version += 1
        for s in obs.sightings:
            if offset is not None:
                s.pos = (s.pos[0] + offset[0], s.pos[1] + offset[1])
            self.sightings[s.sid] = s
            self.version += 1

    def merge(self, other: "Knowledge") -> None:
        self.seen |= other.seen
        self.passable |= other.passable
        new = (other.exits - self.exits) or (other.sightings.keys() - self.sightings.keys())
        self.exits |= other.exits
        self.sightings.update(other.sightings)
        if new:
            self.version += 1

    def weight(self, s: Sighting, t: int) -> float:
        return s.conf * 0.5 ** ((t - s.t) / self.dcfg.evidence_halflife)

    def alarm(self, pos: tuple[int, int], t: int) -> float:
        """Decayed sighting confidence within the alert radius of pos."""
        total = 0.0
        for s in self.sightings.values():
            if np.hypot(s.pos[0] - pos[0], s.pos[1] - pos[1]) <= self.dcfg.alert_radius:
                total += self.weight(s, t)
        return total

    def risk(self, t: int) -> np.ndarray:
        """Predator-risk heatmap: Gaussian bumps at sightings, fading with age."""
        risk = np.zeros((self.size, self.size))
        sig = self.dcfg.risk_sigma
        r = int(3 * sig)
        for s in self.sightings.values():
            w = self.weight(s, t)
            if w < 0.02:
                continue
            r0, c0 = int(round(s.pos[0])), int(round(s.pos[1]))
            ra, rb = max(0, r0 - r), min(self.size, r0 + r + 1)
            ca, cb = max(0, c0 - r), min(self.size, c0 + r + 1)
            if ra >= rb or ca >= cb:
                continue
            yy, xx = np.mgrid[ra:rb, ca:cb]
            risk[ra:rb, ca:cb] += w * np.exp(-((yy - s.pos[0]) ** 2 + (xx - s.pos[1]) ** 2) / (2 * sig ** 2))
        return risk

"""Rule-based hiker policy. Reads only the hiker's own state and its Knowledge, never the World."""
from functools import lru_cache

import numpy as np

from .config import DecisionConfig, HikerConfig
from .knowledge import Knowledge
from .perception import Sighting
from .world import NEIGHBOURS, distance_field, step_down

UNKNOWN_COST = 1.3


@lru_cache(maxsize=4)
def _edge_mask(n: int) -> np.ndarray:
    edge = np.zeros((n, n), bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    edge.flags.writeable = False
    return edge


class Planner:
    def __init__(self, hcfg: HikerConfig, dcfg: DecisionConfig):
        self.hcfg, self.dcfg = hcfg, dcfg
        self._cache: dict[int, tuple[tuple, np.ndarray, np.ndarray]] = {}

    # --- decisions -------------------------------------------------------------------------
    def wants_escape(self, kn: Knowledge, pos: tuple[int, int], t: int) -> bool:
        return kn.alarm(pos, t) >= self.dcfg.escape_threshold

    def wants_scan(self, kn: Knowledge, hiker: int, pos: tuple[int, int], t: int, last_scan: int) -> bool:
        """A teammate reported the predator near me recently, and I haven't looked for it myself."""
        if t - last_scan < self.dcfg.scan_cooldown:
            return False
        return any(s.hiker != hiker and t - s.t <= self.dcfg.scan_window and s.conf > 0
                   and np.hypot(s.pos[0] - pos[0], s.pos[1] - pos[1]) <= self.dcfg.alert_radius
                   for s in kn.sightings.values())

    def relook_target(self, kn: Knowledge, hiker: int) -> Sighting | None:
        lo, hi = self.dcfg.relook_band
        mine = [s for s in kn.sightings.values() if s.hiker == hiker and not s.relooked and lo <= s.conf <= hi]
        return max(mine, key=lambda s: s.t) if mine else None

    # --- movement --------------------------------------------------------------------------
    def wander(self, kn: Knowledge, pos: tuple[int, int], heading: int, rng: np.random.Generator) -> tuple[tuple[int, int], int]:
        if rng.random() < self.hcfg.wander_turn_prob:
            heading = int(rng.integers(len(NEIGHBOURS)))
        options = [heading] + list(rng.permutation(len(NEIGHBOURS)))
        for h in options:
            nxt = (pos[0] + NEIGHBOURS[h][0], pos[1] + NEIGHBOURS[h][1])
            if 0 <= nxt[0] < kn.size and 0 <= nxt[1] < kn.size and kn.passable[nxt]:
                return nxt, int(h)
        return pos, heading

    def navigate(self, kn: Knowledge, pos: tuple[int, int], t: int) -> tuple[int, int]:
        field, walkable = self._field(kn, t)
        return step_down(field, walkable, pos)

    def _field(self, kn: Knowledge, t: int) -> tuple[np.ndarray, np.ndarray]:
        key = (kn.version, t // self.dcfg.replan_every)
        hit = self._cache.get(id(kn))
        if hit and hit[0] == key:
            return hit[1], hit[2]
        known_wall = kn.seen & ~kn.passable
        cost = np.where(known_wall, np.inf, np.where(kn.seen, 1.0, UNKNOWN_COST))
        cost = cost + self.dcfg.risk_weight * kn.risk(t)
        if kn.exits:
            goals = list(kn.exits)
        else:  # exits lie on the forest edge: head for edge cells not yet seen
            goals = _edge_mask(kn.size) & ~kn.seen
            if not goals.any():
                goals = _edge_mask(kn.size) & ~known_wall
        field = distance_field(cost, goals)
        walkable = ~known_wall
        self._cache[id(kn)] = (key, field, walkable)
        return field, walkable

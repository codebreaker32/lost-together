"""The predator: patrol (or ambush near an exit), sense hikers within its radius, pursue, capture."""
from dataclasses import dataclass, field

import numpy as np

from .config import PredatorConfig, PredatorPolicy
from .world import NEIGHBOURS, World, distance_field, step_down


REGION = 8          # chase fields are cached per 8x8 region of the goal
FOLLOW_STEPS = 6    # steps to follow a field before trying greedy again


def dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


@dataclass
class Predator:
    pos: tuple[int, int]
    cfg: PredatorConfig
    home: tuple[int, int]                 # patrol centre (an exit when ambushing)
    state: str = "patrol"                 # patrol | pursue | handling
    target: tuple[int, int] | None = None
    unseen_for: int = 0
    handling_left: int = 0
    budget: float = 0.0
    _follow: int = 0
    _field: np.ndarray | None = field(default=None, repr=False)
    _fields: dict = field(default_factory=dict, repr=False)

    @classmethod
    def spawn(cls, world: World, cfg: PredatorConfig, rng: np.random.Generator) -> "Predator":
        cells = np.argwhere(world.passable)
        far = cells[np.hypot(*(cells - world.trailhead).T) >= cfg.min_start_dist]
        start = tuple(int(v) for v in far[rng.integers(len(far))])
        home = world.exits[rng.integers(len(world.exits))] if cfg.policy == PredatorPolicy.AMBUSH else start
        return cls(pos=start, cfg=cfg, home=home)

    def sense(self, hikers: dict[int, tuple[int, int]], visibility: float, rng: np.random.Generator) -> None:
        """Sense alive hikers; each is missed independently with miss_prob."""
        if self.state == "handling":
            return
        radius = min(self.cfg.sense_radius, visibility) if self.cfg.fog_limits_sensing else self.cfg.sense_radius
        sensed = [(dist(self.pos, p), p) for p in hikers.values()
                  if dist(self.pos, p) <= radius and rng.random() >= self.cfg.miss_prob]
        if sensed:
            self.state, self.target, self.unseen_for = "pursue", min(sensed)[1], 0
        elif self.state == "pursue":
            self.unseen_for += 1
            if self.unseen_for > self.cfg.lose_track_steps:
                self.state, self.target = "patrol", None

    def move(self, world: World, rng: np.random.Generator) -> None:
        if self.state == "handling":
            self.handling_left -= 1
            if self.handling_left <= 0:
                self.state = "patrol"
            return
        self.budget += self.cfg.speed_ratio
        while self.budget >= 1.0:
            self.budget -= 1.0
            if self.state == "patrol" and (self.target is None or self.pos == self.target):
                self.target = self._patrol_point(world, rng)
            self.pos = self._step_toward(world, self.target)

    def captured(self, hiker_pos: tuple[int, int]) -> bool:
        return self.state != "handling" and dist(self.pos, hiker_pos) <= self.cfg.capture_radius

    def start_handling(self) -> None:
        self.state, self.handling_left, self.target = "handling", self.cfg.handling_steps, None

    def _patrol_point(self, world: World, rng: np.random.Generator) -> tuple[int, int]:
        spread = 6 if self.cfg.policy == PredatorPolicy.AMBUSH else world.size
        cells = np.argwhere(world.passable)
        near = cells[np.max(np.abs(cells - self.home), axis=1) <= spread]
        return tuple(int(v) for v in near[rng.integers(len(near))])

    def _step_toward(self, world: World, goal: tuple[int, int]) -> tuple[int, int]:
        """Greedy step toward the goal. When greedy is blocked, follow a shortest-path field to the
        goal's 8x8 region for a few steps. Fields are cached per region, so a chase costs few searches."""
        if self._follow > 0 and self._field is not None:
            self._follow -= 1
            nxt = step_down(self._field, world.passable, self.pos)
            if nxt != self.pos:
                return nxt
        n = world.size
        best, best_d = self.pos, dist(self.pos, goal)
        for dr, dc in NEIGHBOURS:
            r, c = self.pos[0] + dr, self.pos[1] + dc
            if 0 <= r < n and 0 <= c < n and world.passable[r, c] and dist((r, c), goal) < best_d - 1e-9:
                best, best_d = (r, c), dist((r, c), goal)
        if best != self.pos or best_d < 1:
            return best
        self._field, self._follow = self._region_field(world, goal), FOLLOW_STEPS
        return step_down(self._field, world.passable, self.pos)

    def _region_field(self, world: World, goal: tuple[int, int]) -> np.ndarray:
        key = (goal[0] // REGION, goal[1] // REGION)
        if key not in self._fields:
            cells = np.argwhere(world.passable)
            centre = np.array(key) * REGION + REGION // 2
            target = tuple(int(v) for v in cells[np.argmin(np.hypot(*(cells - centre).T))])
            self._fields[key] = distance_field(np.where(world.passable, 1.0, np.inf), [target])
        return self._fields[key]

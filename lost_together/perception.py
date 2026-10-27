"""What a hiker sees. This is the only path from ground truth into a hiker's head.

Observations carry no truth labels. Whether a sighting was real is logged separately
(TruthLog) for the evaluator only.

Two models:
  PERFECT    reference: the predator is seen exactly whenever it is in view and line of sight; no false cues.
  SURROGATE  detection falls off with distance and fog, positions are noisy, false cues appear.
             Its parameters are placeholders until calibrated on the real OpenCV pipeline (Tier A).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Perception, PerceptionConfig
from .world import World, line_of_sight


@dataclass
class Sighting:
    sid: int
    pos: tuple[float, float]
    conf: float
    t: int
    hiker: int
    relooked: bool = False


@dataclass
class Observation:
    hiker: int
    t: int
    cells: tuple[np.ndarray, np.ndarray]   # cells now in view (row, col index arrays)
    passable: np.ndarray                   # passability of those cells
    exits: list[tuple[int, int]]
    sightings: list[Sighting]


@dataclass
class TruthLog:
    """Evaluator-only record of which sightings were real."""
    real: dict[int, bool] = field(default_factory=dict)
    next_id: int = 0

    def new(self, is_real: bool) -> int:
        sid, self.next_id = self.next_id, self.next_id + 1
        self.real[sid] = is_real
        return sid


def visibility(cfg: PerceptionConfig, view_radius: float) -> float:
    if cfg.mode == Perception.CALIBRATED:
        return view_radius if cfg.beta <= 0 else min(view_radius, 3.912 / cfg.beta / cfg.cell_m)
    return view_radius * (1.0 - 0.75 * cfg.fog)


class DetectionTable:
    """P(detect), confidence quantiles and false-cue rates measured with YOLOX on composited forest frames
    (lost_together/vision/calibrate.py), interpolated over distance (m) and fog extinction β (1/m)."""

    def __init__(self, data: dict, threshold: str):
        p = data["p_detect"]
        p = p[threshold] if threshold in p else p
        self.betas = np.array(sorted(float(b) for b in p))
        self.dists = np.array(sorted(float(z) for z in next(iter(p.values()))))
        self.p = np.array([[_get(_get(p, b), z) for z in self.dists] for b in self.betas])
        cq = data.get("conf_quantiles", {})
        self.q = np.array([[_get(_get(cq, b), z) if cq else [0.3, 0.5, 0.7] for z in self.dists] for b in self.betas])
        self.false_bear = np.array([float(_get(data.get("false_bear_per_frame", {}), b, 0.0)) for b in self.betas])
        self.decoy = np.array([float(_get(data.get("decoy_as_bear", {}), b, 0.0)) for b in self.betas])

    @classmethod
    def load(cls, path: str, threshold: str) -> "DetectionTable":
        key = (str(Path(path).resolve()), threshold)
        if key not in _TABLES:
            _TABLES[key] = cls(json.loads(Path(path).read_text()), threshold)
        return _TABLES[key]

    def _interp(self, grid: np.ndarray, d_m: float, beta: float):
        rows = np.array([np.interp(d_m, self.dists, g) if g.ndim == 1 else
                         np.array([np.interp(d_m, self.dists, g[:, k]) for k in range(g.shape[1])]) for g in grid])
        if rows.ndim == 1:
            return float(np.interp(beta, self.betas, rows))
        return np.array([np.interp(beta, self.betas, rows[:, k]) for k in range(rows.shape[1])])

    def p_detect(self, d_m: float, beta: float) -> float:
        return float(np.clip(self._interp(self.p, d_m, beta), 0.0, 1.0))

    def sample_conf(self, d_m: float, beta: float, rng: np.random.Generator) -> float:
        q10, q50, q90 = self._interp(self.q, d_m, beta)
        u = rng.random()   # piecewise-linear inverse CDF through (0.1, q10), (0.5, q50), (0.9, q90)
        return float(np.clip(np.interp(u, [0.0, 0.1, 0.5, 0.9, 1.0], [q10 * 0.8, q10, q50, q90, min(1.0, q90 * 1.05)]), 0.01, 0.99))

    def false_rate(self, beta: float, decoy_encounter_rate: float) -> float:
        return float(np.interp(beta, self.betas, self.false_bear) + decoy_encounter_rate * np.interp(beta, self.betas, self.decoy))


_TABLES: dict = {}


def _get(d: dict, key: float, default=None):
    """Look up a float key in a JSON dict whose keys are strings like '0.02' or '10'."""
    for k, v in d.items():
        if abs(float(k) - key) < 1e-9:
            return v
    if default is not None:
        return default
    raise KeyError(key)


def _disk(radius: float) -> np.ndarray:
    r = int(np.ceil(radius))
    rr, cc = np.mgrid[-r:r + 1, -r:r + 1]
    keep = rr ** 2 + cc ** 2 <= radius ** 2
    return np.stack([rr[keep], cc[keep]], axis=1)


class Perceiver:
    def __init__(self, cfg: PerceptionConfig, view_radius: float, truth: TruthLog):
        self.cfg = cfg
        self.vis = visibility(cfg, view_radius)
        self.offsets = _disk(self.vis)
        self.truth = truth
        self.table = DetectionTable.load(cfg.calibration, cfg.score_threshold) if cfg.mode == Perception.CALIBRATED else None
        zoom = Path(cfg.zoom_calibration)
        self.zoom = DetectionTable.load(str(zoom), cfg.score_threshold) if self.table is not None and zoom.exists() else None

    def observe(self, world: World, hiker: int, pos: tuple[int, int], t: int,
                predator_pos: tuple[int, int] | None, rng: np.random.Generator) -> Observation:
        n = world.size
        cells = self.offsets + pos
        cells = cells[(cells >= 0).all(axis=1) & (cells < n).all(axis=1)]
        rr, cc = cells[:, 0], cells[:, 1]
        exits = [e for e in world.exits if np.hypot(e[0] - pos[0], e[1] - pos[1]) <= self.vis]
        sightings: list[Sighting] = []

        if predator_pos is not None:
            d = float(np.hypot(predator_pos[0] - pos[0], predator_pos[1] - pos[1]))
            if d <= self.vis and line_of_sight(world.passable, pos, predator_pos):
                if self.cfg.mode == Perception.PERFECT:
                    sightings.append(Sighting(self.truth.new(True), tuple(map(float, predator_pos)), 1.0, t, hiker))
                elif self.table is not None:
                    d_m = d * self.cfg.cell_m
                    if rng.random() < self.table.p_detect(d_m, self.cfg.beta):
                        err = rng.normal(0.0, self.cfg.position_noise * max(d, 1.0), size=2)
                        conf = self.table.sample_conf(d_m, self.cfg.beta, rng)
                        sightings.append(Sighting(self.truth.new(True), tuple(np.add(predator_pos, err)), conf, t, hiker))
                else:
                    p = self.cfg.p_detect_near * np.exp(-d / max(self.vis, 1e-6))
                    if rng.random() < p:
                        err = rng.normal(0.0, self.cfg.position_noise * max(d, 1.0), size=2)
                        conf = float(np.clip(rng.beta(4, 2) * (1.0 - 0.5 * d / self.vis), 0.05, 0.99))
                        sightings.append(Sighting(self.truth.new(True), tuple(np.add(predator_pos, err)), conf, t, hiker))

        false_rate = (self.table.false_rate(self.cfg.beta, self.cfg.false_cue_rate) if self.table is not None
                      else self.cfg.false_cue_rate if self.cfg.mode == Perception.SURROGATE else 0.0)
        if false_rate > 0 and rng.random() < false_rate:
            off = self.offsets[rng.integers(len(self.offsets))]
            fake = (pos[0] + off[0], pos[1] + off[1])
            sightings.append(Sighting(self.truth.new(False), tuple(map(float, fake)), float(rng.uniform(0.15, 0.6)), t, hiker))

        return Observation(hiker, t, (rr, cc), world.passable[rr, cc], exits, sightings)

    def scan(self, world: World, hiker: int, pos: tuple[int, int], t: int,
             predator_pos: tuple[int, int] | None, rng: np.random.Generator) -> list[Sighting]:
        """Spend a step scanning with the 3x zoom (prompted by a teammate's report). Calibrated mode only."""
        if self.zoom is None or predator_pos is None:
            return []
        d = float(np.hypot(predator_pos[0] - pos[0], predator_pos[1] - pos[1]))
        if d > self.vis or not line_of_sight(world.passable, pos, predator_pos):
            return []
        d_m = d * self.cfg.cell_m
        if rng.random() >= self.zoom.p_detect(d_m, self.cfg.beta):
            return []
        err = rng.normal(0.0, self.cfg.position_noise * max(d, 1.0), size=2)
        return [Sighting(self.truth.new(True), tuple(np.add(predator_pos, err)),
                         self.zoom.sample_conf(d_m, self.cfg.beta, rng), t, hiker)]

    def relook(self, s: Sighting, rng: np.random.Generator, true_dist: float | None = None) -> Sighting:
        """Spend a step looking again. Real cues firm up; false cues are usually discarded.

        Calibrated: the detector re-runs on a 3x zoom tile, so confirmation follows the measured zoom table at
        the predator's current distance, and a false cue survives only at the zoom false-positive rate."""
        s.relooked = True
        if self.cfg.mode == Perception.PERFECT:
            return s
        if self.zoom is not None:
            d_m = None if true_dist is None else true_dist * self.cfg.cell_m
            if self.truth.real[s.sid] and d_m is not None and rng.random() < self.zoom.p_detect(d_m, self.cfg.beta):
                s.conf = max(s.conf, self.zoom.sample_conf(d_m, self.cfg.beta, rng))
            elif self.truth.real[s.sid]:
                s.conf *= 0.5
            elif rng.random() >= self.zoom.false_rate(self.cfg.beta, 0.0):
                s.conf = 0.0
            return s
        if self.truth.real[s.sid]:
            s.conf = s.conf + (1.0 - s.conf) * self.cfg.relook_gain
        elif rng.random() < self.cfg.relook_gain:
            s.conf = 0.0
        return s

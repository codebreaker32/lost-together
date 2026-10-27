"""Episode configuration. Units: distances in grid cells, time in steps.

Defaults: a 80x80 forest of 5 m cells (400 m square), 1 step = 4 s at walking pace,
so a hiker moves 1 cell per step and T_max = 600 steps is 40 minutes.
"""
from dataclasses import dataclass, field
from enum import Enum


class Cooperation(str, Enum):
    SOLO = "solo"              # each hiker keeps its own map
    SIGHTINGS = "sightings"    # only predator sightings are shared; each hiker maps terrain and exits alone
    MAP = "map"                # only terrain and exits are shared; sightings stay private
    SHARE = "share"            # one living map (terrain, exits, sightings); hikers move independently
    GROUP = "group"            # shared map, and the group moves together once a majority decides to escape


class Perception(str, Enum):
    PERFECT = "perfect"        # perfect-perception reference: exact sightings in view, no false cues
    SURROGATE = "surrogate"    # placeholder noise model (pilots only)
    CALIBRATED = "calibrated"  # detection table measured with YOLOX on composited TartanGround frames


class PredatorPolicy(str, Enum):
    PURSUE = "pursue"      # patrol, then chase the nearest hiker it senses
    AMBUSH = "ambush"      # wait near an exit, chase only what it senses


@dataclass(frozen=True)
class WorldConfig:
    size: int = 80
    obstacle_fraction: float = 0.22
    smoothing_passes: int = 3
    n_exits: int = 3


@dataclass(frozen=True)
class HikerConfig:
    n: int = 4
    speed: float = 1.0
    view_radius: float = 8.0          # clear-air visibility (40 m)
    wander_turn_prob: float = 0.15
    enjoy_steps: int = 150            # after this the hikers head home even without a threat


@dataclass(frozen=True)
class PredatorConfig:
    speed_ratio: float = 1.0          # predator speed / hiker speed
    sense_radius: float = 20.0
    miss_prob: float = 0.3            # per hiker per step, independent, so groups are easier to sense
    capture_radius: float = 1.0
    handling_steps: int = 30          # after a capture the predator pauses
    lose_track_steps: int = 10
    min_start_dist: float = 30.0      # from the trailhead
    policy: PredatorPolicy = PredatorPolicy.PURSUE
    fog_limits_sensing: bool = False


@dataclass(frozen=True)
class PerceptionConfig:
    mode: Perception = Perception.SURROGATE
    fog: float = 0.0                  # 0 clear .. 1 dense; visibility = view_radius * (1 - 0.75 * fog)
    p_detect_near: float = 0.9
    false_cue_rate: float = 0.01      # per hiker per step
    position_noise: float = 0.15      # sighting position error as a fraction of distance
    drift_per_cell: float = 0.05      # localisation drift (cells, std per sqrt(cell walked)) on shared reports
    relook_gain: float = 0.6          # a look-again moves a real cue's confidence this far toward 1,
                                      # and discards a false cue with this probability
    beta: float = 0.0                 # CALIBRATED only: fog extinction per metre (visibility ≈ 3.912/β m)
    cell_m: float = 5.0
    calibration: str = "results/calibration.json"
    zoom_calibration: str = "results/calibration_zoom.json"   # 3x "look closer" tile, same pipeline
    score_threshold: str = "0.25"


@dataclass(frozen=True)
class DecisionConfig:
    alert_radius: float = 15.0
    escape_threshold: float = 0.9     # summed, decayed sighting confidence near the hiker
    evidence_halflife: float = 60.0
    risk_sigma: float = 4.0
    risk_weight: float = 6.0          # planner cost per unit of risk
    relook_band: tuple[float, float] = (0.25, 0.75)
    replan_every: int = 15
    scan_window: int = 10             # a teammate's sighting younger than this (steps) prompts a zoom scan
    scan_cooldown: int = 5


@dataclass(frozen=True)
class CommConfig:
    range_cells: float | None = None  # None = global, instant, lossless sharing


@dataclass(frozen=True)
class EpisodeConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    hikers: HikerConfig = field(default_factory=HikerConfig)
    predator: PredatorConfig = field(default_factory=PredatorConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    cooperation: Cooperation = Cooperation.SOLO
    predator_on: bool = True
    t_max: int = 600

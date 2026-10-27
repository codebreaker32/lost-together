"""One episode: hikers enjoy the forest, notice (or miss) signs of the predator, and try to get out.

Order of play each step: predator senses and moves -> captures -> hikers perceive -> maps update
-> hikers decide -> hikers move -> exits and captures. Ground truth stays in this module and the
metrics; hikers act only on their Knowledge.
"""
from dataclasses import dataclass, field

import numpy as np

from .config import Cooperation, EpisodeConfig
from .knowledge import Knowledge
from .perception import Observation, Perceiver, TruthLog
from .planner import Planner
from .predator import Predator, dist
from .rng import streams
from .world import World, distance_field, generate, step_down

FALSE_ALARM_WINDOW = 20      # steps: an alarm with no predator nearby in this window is a false alarm


@dataclass
class Hiker:
    hid: int
    pos: tuple[int, int]
    heading: int
    kn: Knowledge
    mode: str = "enjoy"          # enjoy | escape | home
    status: str = "active"       # active | escaped | caught
    t_end: int | None = None
    t_decide: int | None = None
    reason: str | None = None    # alarm | time
    drift: np.ndarray = field(default_factory=lambda: np.zeros(2))
    relooks: int = 0
    scans: int = 0
    last_scan: int = -10 ** 9
    last_near: int = -10 ** 9    # evaluator: last step the predator was near this hiker


def run_episode(cfg: EpisodeConfig, seed: int, trace: bool = False) -> dict:
    rs = streams(seed)
    world = generate(cfg.world, rs["world"])
    predator = Predator.spawn(world, cfg.predator, rs["predator"]) if cfg.predator_on else None
    truth = TruthLog()
    perceiver = Perceiver(cfg.perception, cfg.hikers.view_radius, truth)
    planner = Planner(cfg.hikers, cfg.decision)

    n = world.size
    coop = cfg.cooperation
    shared = coop in (Cooperation.SHARE, Cooperation.GROUP)
    sightings_shared = shared or coop == Cooperation.SIGHTINGS
    global_comms = shared and cfg.comm.range_cells is None
    common = Knowledge(n, cfg.decision) if global_comms else None
    hikers = [Hiker(i, world.trailhead, int(rs["spawn"].integers(8)),
                    common if global_comms else Knowledge(n, cfg.decision)) for i in range(cfg.hikers.n)]
    first = hikers[0].kn
    for h in hikers[1:]:              # partial sharing: link only the shared parts of the map
        if coop == Cooperation.SIGHTINGS:
            h.kn.sightings = first.sightings
        elif coop == Cooperation.MAP:
            h.kn.seen, h.kn.passable, h.kn.exits = first.seen, first.passable, first.exits

    group = _Group() if cfg.cooperation == Cooperation.GROUP else None
    near_dist = 2 * perceiver.vis
    closest, threat_onset = np.inf, None
    frames, seen_prev = [], np.zeros((n, n), bool)

    for t in range(cfg.t_max):
        active = [h for h in hikers if h.status == "active"]
        if not active:
            break

        # predator
        if predator:
            predator.sense({h.hid: h.pos for h in active}, perceiver.vis, rs["predator"])
            predator.move(world, rs["predator"])
            _captures(predator, active, t)
            active = [h for h in active if h.status == "active"]
            if not active:
                break
            for h in active:
                d = dist(predator.pos, h.pos)
                closest = min(closest, d)
                if d <= near_dist:
                    h.last_near = t
                if threat_onset is None and d <= cfg.predator.sense_radius:
                    threat_onset = t

        # perceive and update maps
        frame_sightings, events = [], []
        for h in active:
            obs = perceiver.observe(world, h.hid, h.pos, t, predator.pos if predator else None, rs["cues"])
            h.kn.integrate(obs, offset=h.drift.copy() if sightings_shared else None)
            if trace:
                frame_sightings += [(s.sid, round(s.pos[0], 2), round(s.pos[1], 2), round(s.conf, 3), h.hid,
                                     truth.real[s.sid]) for s in obs.sightings]
        if shared and not global_comms:
            _merge_in_range(active, cfg.comm.range_cells)

        # decide
        staying = set()
        for h in active:
            if h.mode != "enjoy":
                continue
            target = planner.relook_target(h.kn, h.hid)
            if target is not None:
                before = target.conf
                perceiver.relook(target, rs["noise"], dist(predator.pos, h.pos) if predator else None)
                h.relooks += 1
                staying.add(h.hid)
                events.append((h.hid, "relook", target.sid, round(before, 3), round(target.conf, 3)))
            elif perceiver.zoom is not None and planner.wants_scan(h.kn, h.hid, h.pos, t, h.last_scan):
                found = perceiver.scan(world, h.hid, h.pos, t, predator.pos if predator else None, rs["cues"])
                h.kn.integrate(Observation(h.hid, t, (np.empty(0, int), np.empty(0, int)), np.empty(0, bool), [], found),
                               offset=h.drift.copy() if sightings_shared else None)
                h.scans, h.last_scan = h.scans + 1, t
                staying.add(h.hid)
                events.append((h.hid, "scan", len(found) > 0))
                if trace:
                    frame_sightings += [(s.sid, round(s.pos[0], 2), round(s.pos[1], 2), round(s.conf, 3), h.hid, True)
                                        for s in found]
            if planner.wants_escape(h.kn, h.pos, t):
                h.mode, h.reason, h.t_decide = "escape", "alarm", t
                events.append((h.hid, "escape", round(h.kn.alarm(h.pos, t), 3)))
            elif t >= cfg.hikers.enjoy_steps:
                h.mode, h.reason, h.t_decide = "home", "time", t
                events.append((h.hid, "home"))

        # move
        if group is not None:
            group.step(active, staying, planner, t, rs["hikers"])
        else:
            for h in active:
                if h.hid in staying:
                    continue
                if h.mode == "enjoy":
                    h.pos, h.heading = planner.wander(h.kn, h.pos, h.heading, rs["hikers"])
                else:
                    h.pos = planner.navigate(h.kn, h.pos, t)
        for h in active:
            h.drift += rs["noise"].normal(0.0, cfg.perception.drift_per_cell, size=2)

        # exits and captures after moving
        for h in active:
            if h.pos in world.exits:
                h.status, h.t_end = "escaped", t
        if predator:
            _captures(predator, [h for h in active if h.status == "active"], t)

        if trace:
            union = np.logical_or.reduce([h.kn.seen for h in hikers])
            frames.append({"t": t,
                           "hikers": [(h.pos, h.status, h.mode) for h in hikers],
                           "predator": (predator.pos, predator.state) if predator else None,
                           "sightings": frame_sightings,
                           "events": events + [(h.hid, h.status) for h in active if h.status != "active"],
                           "seen_new": np.flatnonzero(union & ~seen_prev).tolist()})
            seen_prev = union

    return _metrics(cfg, seed, world, hikers, truth, closest, threat_onset, frames if trace else None)


class _Group:
    """GROUP: once a majority has decided to leave, everyone gathers, then moves as one at the slowest pace."""

    def __init__(self):
        self.phase = "apart"     # apart | gathering | together
        self.point = None
        self._field = None

    def step(self, active, staying, planner: Planner, t: int, rng: np.random.Generator) -> None:
        decided = [h for h in active if h.mode != "enjoy"]
        if self.phase == "apart" and len(decided) * 2 > len(active):
            kn = active[0].kn
            centre = np.mean([h.pos for h in active], axis=0)
            cand = np.argwhere(kn.seen & kn.passable)
            self.point = tuple(int(v) for v in cand[np.argmin(np.hypot(*(cand - centre).T))])
            cost = np.where(kn.seen & ~kn.passable, np.inf, 1.0)
            self._field = distance_field(cost, [self.point])
            self.phase = "gathering"
            for h in active:
                if h.mode == "enjoy":
                    h.mode, h.reason, h.t_decide = "escape", "group", t

        if self.phase == "apart":
            for h in active:
                if h.hid in staying or h.mode != "enjoy":
                    continue          # decided hikers wait for the vote
                h.pos, h.heading = planner.wander(h.kn, h.pos, h.heading, rng)
        elif self.phase == "gathering":
            for h in active:
                if dist(h.pos, self.point) > 1.5:
                    h.pos = step_down(self._field, ~(h.kn.seen & ~h.kn.passable), h.pos)
            if all(dist(h.pos, self.point) <= 1.5 for h in active):
                self.phase = "together"
        if self.phase == "together":
            nxt = planner.navigate(active[0].kn, active[0].pos, t)
            for h in active:
                h.pos = nxt


def _captures(predator: Predator, hikers, t: int) -> None:
    for h in hikers:
        if h.status == "active" and predator.captured(h.pos):
            h.status, h.t_end = "caught", t
            predator.start_handling()
            break


def _merge_in_range(active, rng_cells: float) -> None:
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            if dist(a.pos, b.pos) <= rng_cells:
                a.kn.merge(b.kn)
                b.kn.merge(a.kn)


def _metrics(cfg, seed, world, hikers, truth, closest, threat_onset, frames) -> dict:
    escaped = [h for h in hikers if h.status == "escaped"]
    alarm_decisions = [h for h in hikers if h.reason == "alarm"]
    false_alarms = sum(1 for h in alarm_decisions if h.t_decide - h.last_near > FALSE_ALARM_WINDOW)
    delays = [h.t_decide - threat_onset for h in alarm_decisions if threat_onset is not None and h.t_decide >= threat_onset]
    real = sum(truth.real.values())
    row = {
        "seed": seed,
        "cooperation": cfg.cooperation.value,
        "perception": cfg.perception.mode.value,
        "predator_on": cfg.predator_on,
        "speed_ratio": cfg.predator.speed_ratio,
        "sense_radius": cfg.predator.sense_radius,
        "miss_prob": cfg.predator.miss_prob,
        "fog": cfg.perception.fog,
        "beta": cfg.perception.beta,
        "policy": cfg.predator.policy.value,
        "comm_range": cfg.comm.range_cells or 0,
        "false_cue_rate": cfg.perception.false_cue_rate,
        "n_hikers": len(hikers),
        "frac_escaped": len(escaped) / len(hikers),
        "all_escaped": len(escaped) == len(hikers),
        "n_caught": sum(h.status == "caught" for h in hikers),
        "mean_t_escape": float(np.mean([h.t_end for h in escaped])) if escaped else np.nan,
        "t_end": ";".join(f"{h.t_end if h.t_end is not None else cfg.t_max}:{h.status[0]}" for h in hikers),
        "closest_approach": float(closest),
        "threat_onset": threat_onset if threat_onset is not None else -1,
        "n_alarm_decisions": len(alarm_decisions),
        "n_false_alarms": false_alarms,
        "mean_decision_delay": float(np.mean(delays)) if delays else np.nan,
        "n_relooks": sum(h.relooks for h in hikers),
        "n_scans": sum(h.scans for h in hikers),
        "n_sightings_real": real,
        "n_sightings_false": len(truth.real) - real,
    }
    if frames is not None:
        row["frames"] = frames
        row["world"] = world
    return row

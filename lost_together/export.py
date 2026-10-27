"""Export traced episodes as compact JSON for the replay viewer (viewer/).

  uv run python -m lost_together.export --seed 7 --coops solo share --out viewer/data
  uv run python -m lost_together.export --find-contrast 80 --coops solo share --out viewer/data

Each episode becomes <out>/<episode_id>.json; <out>/index.json lists every exported episode
(merged with what is already there) and suggests a default side-by-side pair.

--find-contrast N searches seeds 0..N-1 for the hero seed: SOLO loses at least one hiker and
SHARE gets everyone out (largest gap first). Ground truth (walls, predator, real/false labels)
is in the export on purpose: the viewer is the evaluator's view, not a hiker's.
"""
import argparse
import base64
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import Cooperation, EpisodeConfig, Perception
from .perception import _disk, visibility
from .sim import run_episode

SCHEMA = 1
STEP_SECONDS = 4          # config.py: 1 step = 4 s
CELL_METRES = 5
LABELS = {
    "solo": "SOLO",
    "sightings": "SHARED SIGHTINGS",
    "map": "SHARED TERRAIN",
    "share": "SHARED MAP",
    "group": "SHARED MAP + GROUP",
}
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "viewer" / "data"
NEVER = 0xFFFF            # own_seen sentinel: cell never seen by that hiker
PRIVATE_MAP = ("solo", "sightings")         # each hiker maps terrain and exits alone
PRIVATE_SIGHTINGS = ("solo", "map")         # each hiker keeps its own predator sightings


# --- encoding helpers ---------------------------------------------------------------------
def encode_mask(mask: np.ndarray) -> str:
    """Row-major bool array -> base64 of a big-endian bitset (np.packbits)."""
    return base64.b64encode(np.packbits(mask.astype(bool).ravel()).tobytes()).decode("ascii")


def decode_mask(b64: str, size: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(b64), np.uint8))[: size * size]
    return bits.astype(bool).reshape(size, size)


def _clean(v):
    """JSON-safe: numpy scalars -> python, NaN/inf -> None, tuples -> lists."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return round(f, 4) if math.isfinite(f) else None
    return v


PERCEPTION: dict = {}   # CLI override, e.g. {"mode": Perception.CALIBRATED, "beta": 0.05}; inherited by workers


def make_config(coop: str, speed_ratio: float, sense_radius: float, t_max: int | None = None,
                fog: float | None = None) -> EpisodeConfig:
    base = EpisodeConfig()
    cfg = replace(base, cooperation=Cooperation(coop),
                  predator=replace(base.predator, speed_ratio=speed_ratio, sense_radius=sense_radius))
    if PERCEPTION:
        cfg = replace(cfg, perception=replace(cfg.perception, **PERCEPTION))
    if fog is not None:
        cfg = replace(cfg, perception=replace(cfg.perception, fog=fog))
    if t_max is not None:
        cfg = replace(cfg, t_max=t_max)
    return cfg


def episode_id(seed: int, cfg: EpisodeConfig) -> str:
    eid = f"seed{seed}-{cfg.cooperation.value}-sr{cfg.predator.speed_ratio:g}-R{cfg.predator.sense_radius:g}"
    if cfg.perception.fog:
        eid += f"-fog{cfg.perception.fog:g}"
    if cfg.perception.mode == Perception.CALIBRATED:
        eid += f"-cal-b{cfg.perception.beta:g}"
    return eid


def encode_u16(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<u2").tobytes()).decode("ascii")


def decode_u16(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype="<u2")


def own_first_seen(frames: list[dict], trailhead, n: int, vis: float) -> np.ndarray:
    """Per-hiker first-seen step of every cell, rebuilt from the exported frames (n_hikers x n*n, NEVER if unseen).

    sim.py's trace only logs the union of the hikers' maps. A hiker observes, at the start of step t, the full
    disk of radius `vis` around where it ended step t-1 (perception.Perceiver.observe), provided it is still
    active after the predator's move; a capture at the start of step t leaves the hiker where it stood.
    """
    nh = len(frames[0]["h"])
    first = np.full((nh, n * n), NEVER, np.uint16)
    disk = _disk(vis)
    prev_pos = [tuple(trailhead)] * nh
    prev_status = ["active"] * nh
    for f in frames:
        t = f["t"]
        for i, (r, c, status, _mode) in enumerate(f["h"]):
            if prev_status[i] == "active":
                caught_at_start = status == "caught" and (r, c) == prev_pos[i]
                if not caught_at_start:
                    cells = disk + prev_pos[i]
                    cells = cells[(cells >= 0).all(axis=1) & (cells < n).all(axis=1)]
                    idx = cells[:, 0] * n + cells[:, 1]
                    first[i, idx] = np.minimum(first[i, idx], t)
            prev_pos[i], prev_status[i] = (r, c), status
    return first


# --- episode -> dict ----------------------------------------------------------------------
def encode_episode(row: dict, cfg: EpisodeConfig) -> dict:
    """Turn a traced run_episode row into the viewer's JSON structure."""
    world, frames = row["world"], row["frames"]
    n_hikers = row["n_hikers"]
    metrics = {k: v for k, v in row.items() if k not in ("frames", "world")}

    # Final status per hiker from the metrics (the trace misses a capture that ends the episode
    # before the step's frame is written).
    final = []
    for part in row["t_end"].split(";"):
        t, s = part.split(":")
        final.append((int(t), {"e": "escaped", "c": "caught", "a": "active"}[s]))

    out_frames, prev_status = [], ["active"] * n_hikers
    pred_prev = None
    for f in frames:
        events = []
        for e in f["events"]:
            hid, kind = int(e[0]), e[1]
            if kind in ("escaped", "caught"):
                continue                      # re-derived below from status changes (complete)
            events.append([hid, kind, *e[2:]])
        for i, (pos, status, mode) in enumerate(f["hikers"]):
            if status != prev_status[i]:
                events.append([i, status])
                prev_status[i] = status
        pred = f["predator"]
        if pred is not None and pred[1] != pred_prev:
            if pred_prev is not None:
                events.append([-1, "predator", pred[1]])
            pred_prev = pred[1]
        out_frames.append({
            "t": int(f["t"]),
            "h": [[int(p[0]), int(p[1]), status, mode] for p, status, mode in f["hikers"]],
            "p": [int(pred[0][0]), int(pred[0][1]), pred[1]] if pred is not None else None,
            "s": [[int(sid), r, c, conf, int(h), bool(real)] for sid, r, c, conf, h, real in f["sightings"]],
            "e": events,
            "n": [int(x) for x in f["seen_new"]],
        })

    # Close the trace: if a hiker's final status never appeared, add a frame at its end time.
    missing = [(t, i, s) for i, (t, s) in enumerate(final) if s != "active" and prev_status[i] != s]
    if missing and out_frames:
        last = out_frames[-1]
        t_close = max(max(t for t, _, _ in missing), last["t"] + 1)
        hikers = [list(h) for h in last["h"]]
        events = []
        for _, i, s in missing:
            hikers[i][2] = s
            events.append([i, s])
        out_frames.append({"t": t_close, "h": hikers, "p": last["p"], "s": [], "e": events, "n": []})

    vis = visibility(cfg.perception, cfg.hikers.view_radius)
    world_json = {
        "size": world.size,
        "passable": encode_mask(world.passable),
        "exits": [list(e) for e in world.exits],
        "trailhead": list(world.trailhead),
    }
    if cfg.cooperation.value in PRIVATE_MAP and out_frames:
        first = own_first_seen(out_frames, world.trailhead, world.size, vis)
        world_json["own_seen"] = [encode_u16(a) for a in first]

    return _clean({
        "schema": SCHEMA,
        "id": episode_id(row["seed"], cfg),
        "seed": row["seed"],
        "cooperation": cfg.cooperation.value,
        "label": LABELS[cfg.cooperation.value],
        "config": {
            "speed_ratio": cfg.predator.speed_ratio,
            "sense_radius": cfg.predator.sense_radius,
            "capture_radius": cfg.predator.capture_radius,
            "miss_prob": cfg.predator.miss_prob,
            "handling_steps": cfg.predator.handling_steps,
            "view_radius": cfg.hikers.view_radius,
            "vis": vis,
            "private_map": cfg.cooperation.value in PRIVATE_MAP,
            "private_sightings": cfg.cooperation.value in PRIVATE_SIGHTINGS,
            "enjoy_steps": cfg.hikers.enjoy_steps,
            "n_hikers": cfg.hikers.n,
            "fog": cfg.perception.fog,
            "perception": cfg.perception.mode.value,
            "false_cue_rate": cfg.perception.false_cue_rate,
            "alert_radius": cfg.decision.alert_radius,
            "escape_threshold": cfg.decision.escape_threshold,
            "evidence_halflife": cfg.decision.evidence_halflife,
            "risk_sigma": cfg.decision.risk_sigma,
            "t_max": cfg.t_max,
            "step_seconds": STEP_SECONDS,
            "cell_metres": CELL_METRES,
        },
        "metrics": metrics,
        "final": [{"t": t, "status": s} for t, s in final],
        "world": world_json,
        "frames": out_frames,
    })


def summary(ep: dict, out_dir: Path) -> dict:
    m = ep["metrics"]
    entry = {
        "id": ep["id"], "file": f"{ep['id']}.json", "seed": ep["seed"], "cooperation": ep["cooperation"],
        "label": ep["label"], "speed_ratio": ep["config"]["speed_ratio"], "sense_radius": ep["config"]["sense_radius"],
        "fog": ep["config"]["fog"], "n_hikers": m["n_hikers"], "frac_escaped": m["frac_escaped"],
        "n_caught": m["n_caught"], "n_frames": len(ep["frames"]),
    }
    cam = camera_listing(out_dir, ep["id"])
    if cam:
        entry["camera"] = cam
    return entry


def camera_listing(out_dir: Path, eid: str) -> dict[str, list[int]]:
    """Camera-tile hook: viewer/data/frames/<episode_id>/<t>_<hiker>.jpg -> {hiker: [t, ...]}."""
    fdir = out_dir / "frames" / eid
    tiles: dict[str, list[int]] = {}
    if fdir.is_dir():
        for p in fdir.glob("*.jpg"):
            try:
                t, h = p.stem.split("_")
                tiles.setdefault(str(int(h)), []).append(int(t))
            except ValueError:
                continue
    return {h: sorted(ts) for h, ts in tiles.items()}


def write_index(out_dir: Path, new_entries: list[dict], default_pair: tuple[str, str] | None) -> dict:
    path = out_dir / "index.json"
    index = {"schema": SCHEMA, "episodes": [], "default": None}
    if path.exists():
        try:
            index = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    by_id = {e["id"]: e for e in index.get("episodes", [])}
    for e in new_entries:
        by_id[e["id"]] = e
    episodes = []
    for e in by_id.values():             # drop entries whose file vanished; refresh camera listings
        if not (out_dir / e["file"]).exists():
            continue
        e = {k: v for k, v in e.items() if k != "camera"}
        cam = camera_listing(out_dir, e["id"])
        if cam:
            e["camera"] = cam
        episodes.append(e)
    order = list(LABELS)
    episodes.sort(key=lambda e: (e["seed"], e["speed_ratio"], e["sense_radius"], e.get("fog", 0.0),
                                 order.index(e["cooperation"])))
    index["schema"] = SCHEMA
    index["episodes"] = episodes
    ids = {e["id"] for e in episodes}
    if default_pair is not None:
        index["default"] = {"left": default_pair[0], "right": default_pair[1]}
    elif index.get("default") and not {index["default"].get("left"), index["default"].get("right")} <= ids:
        index["default"] = None
    if not index.get("default") and len(episodes) >= 2:
        index["default"] = {"left": episodes[0]["id"], "right": episodes[-1]["id"]}
    path.write_text(json.dumps(index, indent=1))
    return index


# --- hero seed search ---------------------------------------------------------------------
def _outcome(job):
    seed, coop, sr, rad, fog = job
    row = run_episode(make_config(coop, sr, rad, fog=fog), seed)
    return seed, coop, row["frac_escaped"], row["n_caught"], row["all_escaped"]


def find_contrast(n_seeds: int, speed_ratio: float, sense_radius: float, fog: float | None = None,
                  start: int = 0, workers: int | None = None, verbose: bool = True) -> int | None:
    """Seed in [start, start+n) where SOLO loses someone and SHARE gets everyone out.

    Ranked by escape-rate gap, then by hikers SOLO lost to the predator, then lowest seed.
    """
    jobs = [(s, c, speed_ratio, sense_radius, fog) for s in range(start, start + n_seeds) for c in ("solo", "share")]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    res: dict[int, dict] = {}
    with ProcessPoolExecutor(workers) as pool:
        for seed, coop, frac, caught, all_out in pool.map(_outcome, jobs, chunksize=2):
            res.setdefault(seed, {})[coop] = (frac, caught, all_out)
    hits = []
    for seed, r in res.items():
        (sf, sc, _), (hf, _, h_all) = r["solo"], r["share"]
        if sf < 1.0 and h_all:
            hits.append((-(hf - sf), -sc, seed))
    hits.sort()
    if verbose:
        n_solo_lost = sum(1 for r in res.values() if r["solo"][0] < 1)
        print(f"searched {n_seeds} seeds: solo lost someone in {n_solo_lost}, contrast seeds: "
              f"{[h[2] for h in hits[:10]]}{' ...' if len(hits) > 10 else ''}")
    return hits[0][2] if hits else None


# --- CLI ----------------------------------------------------------------------------------
def export(seed: int, coops: list[str], out_dir: Path, speed_ratio: float = 1.0, sense_radius: float = 20.0,
           fog: float | None = None, t_max: int | None = None, default_pair: tuple[str, str] | None = None,
           verbose: bool = True) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for coop in coops:
        cfg = make_config(coop, speed_ratio, sense_radius, t_max=t_max, fog=fog)
        ep = encode_episode(run_episode(cfg, seed, trace=True), cfg)
        path = out_dir / f"{ep['id']}.json"
        path.write_text(json.dumps(ep, separators=(",", ":")))
        entries.append(summary(ep, out_dir))
        if verbose:
            m = ep["metrics"]
            print(f"{path}  {ep['label']:<20} escaped {m['frac_escaped']:.0%}  caught {m['n_caught']}  "
                  f"frames {len(ep['frames'])}  {path.stat().st_size / 1024:.0f} KB")
    if default_pair is None and len(entries) >= 2:
        default_pair = (entries[0]["id"], entries[-1]["id"])
    write_index(out_dir, entries, default_pair)
    return entries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--coops", nargs="+", default=["solo", "share"], choices=[c.value for c in Cooperation])
    p.add_argument("--speed-ratio", type=float, default=1.0)
    p.add_argument("--sense-radius", type=float, default=20.0)
    p.add_argument("--fog", type=float, default=None)
    p.add_argument("--perception", choices=[m.value for m in Perception], default=None,
                   help="perception model; 'calibrated' uses results/calibration.json (YOLOX on composited frames)")
    p.add_argument("--beta", type=float, default=None, help="fog extinction per metre for calibrated perception")
    p.add_argument("--t-max", type=int, default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--find-contrast", type=int, metavar="N", default=None,
                   help="search seeds 0..N-1 for one where SOLO loses a hiker and SHARE gets everyone out")
    p.add_argument("--left", default=None, help="cooperation mode for the viewer's left map (default: first of --coops)")
    p.add_argument("--right", default=None, help="cooperation mode for the viewer's right map (default: last of --coops)")
    args = p.parse_args(argv)
    if args.perception:
        PERCEPTION["mode"] = Perception(args.perception)
    if args.beta is not None:
        PERCEPTION["beta"] = args.beta

    seed = args.seed
    if args.find_contrast:
        found = find_contrast(args.find_contrast, args.speed_ratio, args.sense_radius, args.fog)
        if found is None:
            raise SystemExit("no contrast seed found; try a larger N or another speed ratio / sense radius")
        seed = found
        print(f"hero seed: {seed}")
    pair = None
    if args.left or args.right:
        def eid(c):
            return episode_id(seed, make_config(c, args.speed_ratio, args.sense_radius, fog=args.fog))
        pair = (eid(args.left or args.coops[0]), eid(args.right or args.coops[-1]))
    export(seed, args.coops, args.out, args.speed_ratio, args.sense_radius, args.fog, args.t_max, pair)
    print(f"index: {args.out / 'index.json'}  ·  view: python -m http.server -d {args.out.parent} 8000")


if __name__ == "__main__":
    main()

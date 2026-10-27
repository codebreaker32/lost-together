"""Tier C input: the moments a hiker had to judge an ambiguous sighting ("look again" escalations).

Runs episodes with tracing and records, for every relook event, what the hiker knew (confidence,
the shared-map summary) and what was true (evaluator only: real or false cue, true distance,
whether escaping was warranted). A later render step turns each point into a camera frame.

  uv run python -m lost_together.escalations --seeds 2000 2060 --out results/escalations_raw.jsonl
"""
import argparse
import json
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import Cooperation, EpisodeConfig, Perception
from .sim import run_episode


def collect(cfg: EpisodeConfig, seed: int) -> list[dict]:
    row = run_episode(cfg, seed, trace=True)
    frames, points = row["frames"], []
    world, n = row["world"], row["world"].size
    seen = np.zeros(n * n, bool)
    sightings, log = {}, []
    for f in frames:
        seen[f["seen_new"]] = True
        for sid, r, c, conf, hiker, real in f["sightings"]:
            sightings[sid] = (r, c, real)
            log.append({"pos": [r, c], "conf": conf, "t": f["t"], "hiker": hiker})
        for ev in f["events"]:
            if len(ev) < 5 or ev[1] != "relook":
                continue
            hiker, _, sid, before, after = ev
            hpos = f["hikers"][hiker][0]
            pred = f["predator"][0] if f["predator"] else None
            r, c, real = sightings[sid]
            true_d = float(np.hypot(pred[0] - hpos[0], pred[1] - hpos[1])) if pred else None
            claimed_d = float(np.hypot(r - hpos[0], c - hpos[1]))
            # what the hiker's (shared) living map held at that moment, in the agent's query_map schema
            summary = {
                "t": f["t"], "cell_m": 5.0, "step_s": 4.0,
                "hiker": {"id": hiker, "pos": list(hpos)},
                "group": [{"id": i, "pos": list(h[0])} for i, h in enumerate(f["hikers"]) if i != hiker and h[1] == "active"],
                "exits": [list(e) for e in world.exits if seen[e[0] * n + e[1]]],
                "sightings": [s for s in log if f["t"] - s["t"] <= 3 * cfg.decision.evidence_halflife],
                "halflife": cfg.decision.evidence_halflife, "alert_radius": cfg.decision.alert_radius,
                "escape_threshold": cfg.decision.escape_threshold,
            }
            points.append({
                "seed": seed, "t": f["t"], "hiker": hiker, "sid": sid, "map_summary": summary,
                "cooperation": cfg.cooperation.value, "beta": cfg.perception.beta,
                "observed": {"conf": before, "distance_cells": round(claimed_d, 2),
                             "bearing_deg": round(float(np.degrees(np.arctan2(c - hpos[1], r - hpos[0]))), 1)},
                "truth": {"predator_real": bool(real),
                          "true_distance_cells": None if true_d is None else round(true_d, 2),
                          "should_escape": bool(real) and true_d is not None
                          and true_d <= cfg.decision.alert_radius,
                          "relook_conf_after": after},
            })
    return points


def render_points(raw: Path, out_dir: Path, limit: int | None = None) -> Path:
    """Turn raw escalation points into Tier C bench input: a composited frame pair per point.

    Real sightings get a bear at the true distance; false cues get a deer or dog decoy at the claimed
    distance, or nothing at all (a shadow). Both real and decoy objects move slightly between the two
    frames, so motion alone does not give the answer away. Fog matches the episode's beta.
    """
    import cv2

    from .vision import calibrate as CAL
    from .vision import composite as C

    bgs = json.loads((C.TG / "backgrounds.json").read_text())
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    pts = [json.loads(line) for line in raw.open()]
    if limit:
        pts = pts[:limit]
    out = out_dir / "points.jsonl"
    with out.open("w") as f:
        for p in pts:
            pid = f"s{p['seed']}_t{p['t']}_h{p['hiker']}_b{p['beta']}"
            rng = np.random.default_rng(zlib.crc32(pid.encode()))  # str hash() is salted per process
            bg = bgs[int(rng.integers(len(bgs)))]
            frame = C.load_frame(bg["traj"], bg["frame"])
            real = p["truth"]["predator_real"]
            if real:
                cid = ["bear_grizzly_rocks", "bear_grizzly_snow"][int(rng.integers(2))]
                z = max(4.0, p["truth"]["true_distance_cells"] * 5.0)
            else:
                cid = [None, "deer_doe_standing", "deer_doe_grazing", "dog_shepherd"][int(rng.integers(4))]
                z = max(4.0, p["observed"]["distance_cells"] * 5.0)
            objs_now, objs_prev = [], []
            if cid:
                bearing = float(CAL.pick_bearings(frame, cid, z, k=1)[0])
                objs_now = [(cid, z, bearing)]
                objs_prev = [(cid, z + float(rng.uniform(0.5, 1.5)), bearing + float(rng.uniform(-2, 2)))]
            now = C.render(frame, objs_now, beta=p["beta"], stereo=False)["left"]
            prev = C.render(frame, objs_prev, beta=p["beta"], stereo=False)["left"]
            cv2.imwrite(str(frames_dir / f"{pid}.jpg"), now, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(frames_dir / f"{pid}_prev.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 92])
            f.write(json.dumps({
                "point_id": pid,
                "frame_path": f"frames/{pid}.jpg",
                "prev_frame_path": f"frames/{pid}_prev.jpg",
                "candidates": None,
                "map_summary": p["map_summary"],
                "truth": {"predator_real": real, "should_escape": p["truth"]["should_escape"],
                          "object": cid or "none"},
            }) + "\n")
    return out


def main(argv=None):
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "render":
        r = argparse.ArgumentParser(prog="escalations render")
        r.add_argument("raw", type=Path)
        r.add_argument("--out", type=Path, default=Path("results/tierC"))
        r.add_argument("--limit", type=int)
        a = r.parse_args(sys.argv[2:])
        print(f"wrote {render_points(a.raw, a.out, a.limit)}")
        return
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs=2, default=[2000, 2060], help="half-open range; disjoint from sweep seeds")
    p.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.05, 0.1])
    p.add_argument("--speed-ratio", type=float, default=1.5)
    p.add_argument("--sense-radius", type=float, default=20)
    p.add_argument("--coop", default="share")
    p.add_argument("--perception", default="calibrated")
    p.add_argument("--calibration", default="results/calibration.json")
    p.add_argument("--out", type=Path, default=Path("results/escalations_raw.jsonl"))
    args = p.parse_args(argv)
    base = EpisodeConfig()
    n = 0
    with args.out.open("w") as f:
        for beta in args.beta:
            cfg = replace(base, cooperation=Cooperation(args.coop),
                          predator=replace(base.predator, speed_ratio=args.speed_ratio, sense_radius=args.sense_radius),
                          perception=replace(base.perception, mode=Perception(args.perception), beta=beta,
                                             calibration=args.calibration))
            for seed in range(*args.seeds):
                for pt in collect(cfg, seed):
                    f.write(json.dumps(pt) + "\n")
                    n += 1
    print(f"wrote {n} escalation points to {args.out}")


if __name__ == "__main__":
    main()

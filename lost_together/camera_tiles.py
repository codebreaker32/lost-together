"""Camera views for the replay viewer: what a hiker's camera saw at the moments that mattered.

For every sighting (real or false) and every "look closer" in an exported episode, render the hiker's
forward camera view with the OpenCV pipeline: a TartanGround forest frame, the bear (real sighting) or a
decoy (false cue) composited at the true metric distance and bearing, fog for the episode's beta, then
YOLOX on the OpenCV 5 DNN engine. A look-closer also gets the zoomed tile the detector re-ran on.
Plus a quiet frame every QUIET_EVERY steps so the tile never goes stale.

Writes viewer/data/frames/<episode_id>/<t>_<hiker>.jpg (raw view, no boxes), zoom/<t>_<hiker>.jpg, and
manifest.json with every detection, so the viewer draws boxes in its own style.

  uv run python -m lost_together.camera_tiles viewer/data/seed26-solo-sr1-R20.json viewer/data/seed26-share-sr1-R20.json
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .vision import calibrate as CAL
from .vision import composite as C
from .vision.detector import YoloxDetector

QUIET_EVERY = 25
ZOOM = 3          # must match results/calibration_zoom.json meta.zoom
TRIES = 5
BEARS = ("bear_grizzly_rocks", "bear_grizzly_snow")
DECOYS = ("deer_doe_standing", "deer_doe_grazing", "dog_shepherd")


def _heading(frames, i, hid):
    """Direction of travel (deg, map frame) from the last few steps."""
    r1, c1 = frames[i]["h"][hid][:2]
    for j in range(i - 1, max(-1, i - 6), -1):
        r0, c0 = frames[j]["h"][hid][:2]
        if (r0, c0) != (r1, c1):
            return math.degrees(math.atan2(c1 - c0, r1 - r0))
    return 0.0


def _rel_bearing(heading, dr, dc):
    b = math.degrees(math.atan2(dc, dr)) - heading
    b = (b + 180) % 360 - 180
    return float(np.clip(b, -38, 38))


def _r(box):
    return [round(float(v), 1) for v in box]


def _dets(dets, truth):
    """Detections with an honest label: does the box actually overlap the object we placed (IoU > 0.3)?"""
    return [{"cls": d.cls, "score": round(d.score, 3), "box": _r(d.box),
             "matches_truth": bool(truth is not None and CAL.iou(d.box, truth) > CAL.IOU_MIN)} for d in dets]


def _view(f, i, hid, kind, r, c, frames, bgs, beta, cell, det, looked, expected, tries=TRIES):
    """Render this hiker's view. The simulator already decided, from the calibrated table, whether the
    predator was seen; pick, among up to `tries` real composites, one whose actual YOLOX output agrees
    (full frame, or the 3x look-closer tile when the hiker zoomed). Disagreement is recorded, never hidden."""
    hr, hc = f["h"][hid][:2]
    dist = None if kind == "quiet" else max(4.0, math.hypot(r - hr, c - hc) * cell)
    view = img = tile = None
    for k in range(tries if kind == "real" else 1):
        bg = bgs[(hid * 7 + f["t"] // 20 + 11 * k) % len(bgs)]
        frame = C.load_frame(bg["traj"], bg["frame"])
        objs = []
        if kind != "quiet":
            pool = BEARS if kind == "real" else DECOYS
            cid = pool[(f["t"] + hid + k) % len(pool)]
            # same placement rule as the calibration table: the most-visible bearing at that distance
            objs = [(cid, dist, float(CAL.pick_bearings(frame, cid, dist, k=1)[0]))]
        rend = C.render(frame, objs, beta=beta, stereo=False)
        img, tile = rend["left"], None
        truth = rend["boxes"][0] if objs else None
        view = {"t": f["t"], "hiker": hid, "kind": kind, "distance_m": None if dist is None else round(dist, 1),
                "background": frame.name, "truth_box": _r(truth) if truth else None,
                "detections": _dets(det.detect(img), truth), "attempt": k + 1}
        if looked and objs:
            pl = rend["placements"][0]
            tile, (x0, y0) = CAL.zoom_tile(img, pl["col"], pl["foot_row"], ZOOM)
            ztruth = ((truth[0] - x0) * ZOOM, (truth[1] - y0) * ZOOM, truth[2] * ZOOM, truth[3] * ZOOM)
            view["zoom"] = {"factor": ZOOM, "origin": [x0, y0], "truth_box": _r(ztruth),
                            "detections": _dets(det.detect(tile), ztruth)}
        seen = any(d["matches_truth"] for d in view["detections"] + view.get("zoom", {}).get("detections", []))
        if kind != "real" or seen == expected:
            break
    if kind == "real":
        view["sim_detected"], view["render_agrees"] = expected, seen == expected
    return view, img, tile


def render_episode(path: Path, det: YoloxDetector, bgs: list) -> dict:
    ep = json.loads(path.read_text())
    eid, frames = ep["id"], ep["frames"]
    beta = float(ep["config"].get("beta") or ep.get("metrics", {}).get("beta") or 0.0)   # export keeps beta in metrics
    cell = float(ep["config"].get("cell_metres", 5))
    out = path.parent / "frames" / eid
    (out / "zoom").mkdir(parents=True, exist_ok=True)
    manifest = {"episode": eid, "beta": beta, "model": "OpenCV Zoo YOLOX-s, cv2.dnn ENGINE_NEW", "views": []}
    looks = {(f["t"], e[0]) for f in frames for e in f["e"] if len(e) >= 2 and e[1] in ("relook", "scan")}
    for i, f in enumerate(frames):
        jobs = {}
        for sid, r, c, conf, hid, real in f["s"]:
            if real and f["p"]:     # the camera saw the actual predator; the sighting position carries noise
                r, c = f["p"][0], f["p"][1]
            jobs[hid] = ("real" if real else "decoy", r, c, True)
        for t, hid in looks:
            if t == f["t"] and hid not in jobs and f["p"]:
                jobs[hid] = ("real", f["p"][0], f["p"][1], False)   # looked, simulator did not detect
        if f["t"] % QUIET_EVERY == 0:
            for hid, h in enumerate(f["h"]):
                if h[2] == "active" and hid not in jobs:
                    jobs[hid] = ("quiet", None, None, False)
        for hid, (kind, r, c, expected) in jobs.items():
            view, img, tile = _view(f, i, hid, kind, r, c, frames, bgs, beta, cell, det, (f["t"], hid) in looks, expected)
            cv2.imwrite(str(out / f"{f['t']}_{hid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 86])
            if tile is not None:
                cv2.imwrite(str(out / "zoom" / f"{f['t']}_{hid}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 86])
            manifest["views"].append(view)
    (out / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("episodes", type=Path, nargs="+")
    args = p.parse_args(argv)
    cv2.setNumThreads(4)
    det = YoloxDetector(conf=0.25)
    bgs = json.loads((C.TG / "backgrounds.json").read_text())
    for path in args.episodes:
        m = render_episode(path, det, bgs)
        n_z = sum("zoom" in v for v in m["views"])
        hits = sum(v.get("render_agrees", False) for v in m["views"] if v["kind"] == "real")
        print(f"{m['episode']}: {len(m['views'])} views ({n_z} look-closer zooms), render agrees with the simulator in "
              f"{hits}/{sum(v['kind'] == 'real' for v in m['views'])} real-predator views")


if __name__ == "__main__":
    main()

"""Calibrate the perception surrogate from pixels: YOLOX (OpenCV 5 DNN, ENGINE_NEW) on composited, fogged
TartanGround frames.

Grid: Z in {5,10,15,20,30,40,60} m x beta in {0,0.02,0.05,0.1,0.2} /m x backgrounds x bear cut-outs x 2
bearings. For each composite we record the best 'bear' score among detections whose box has IoU > 0.3 with
the box of the visible (unoccluded) pasted mask. Bearings are chosen per (background, Z) to be the two most
visible directions in the FOV (the simulator models line of sight itself), and P(detect) is reported for
placements with >= 50% of the object visible. Same placements for every beta (paired design).

Also: clean frames (false 'bear' per frame), decoys (deer, dog at 10-30 m -> rate reported as 'bear'),
hikers ('person' detection rate vs Z, beta), and a compact logistic surrogate p(d, beta).

    uv run python -m lost_together.vision.calibrate [--workers 6] [--n-bg 20]
"""
import argparse
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from . import composite as C
from .cutouts import BEARS, HEIGHT_M, PERSONS, RETIRED, cls_of, load

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DISTANCES = (5, 10, 15, 20, 30, 40, 60)
BETAS = (0.0, 0.02, 0.05, 0.1, 0.2)
THRESHOLDS = (0.25, 0.5)
DECOY_Z = (10, 20, 30)
DECOY_IDS = ("deer_doe_standing", "deer_doe_grazing", "dog_shepherd")
BEARING_CANDIDATES = tuple(range(-38, 39, 4))
CONF = 0.1
IOU_MIN = 0.3
VIS_MIN = 0.5

_DET = None


def visibility(frame: C.Frame, cid: str, z: float, bearing: float, margin: float = 0.15) -> float:
    """Fraction of the cut-out's alpha that would be unoccluded and inside the image (no pasting)."""
    bgra = load(cid)
    hpx = C.F * HEIGHT_M[cls_of(cid)] / z
    ow, oh = max(1, round(bgra.shape[1] * hpx / bgra.shape[0])), max(1, round(hpx))
    alpha = cv2.resize(bgra[..., 3], (ow, oh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    u = C.bearing_to_col(bearing)
    vf, _ = C.foot_row(frame, z, u, ow / 2)
    x0, y0 = int(round(u - ow / 2)), int(round(vf - oh + 1))
    h, w = frame.depth.shape
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + ow, w), min(y0 + oh, h)
    if xa >= xb or ya >= yb:
        return 0.0
    a = alpha[ya - y0:yb - y0, xa - x0:xb - x0]
    vis = a * (frame.depth[ya:yb, xa:xb] > z - margin)
    return float(vis.sum() / max(alpha.sum(), 1e-6))


def pick_bearings(frame, cid, z, k=2, sep=12):
    scored = sorted(((visibility(frame, cid, z, b), b) for b in BEARING_CANDIDATES), reverse=True)
    out = []
    for v, b in scored:
        if all(abs(b - o) >= sep for _, o in out):
            out.append((v, b))
        if len(out) == k:
            break
    return [b for _, b in out]


def mask_box(mask, thr=0.5):
    ys, xs = np.nonzero(mask > thr)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1e-9)


def zoom_tile(img, u, foot_row, zoom):
    """Crop a (640/zoom)^2 tile centred on column u with the ground-contact row at 60% height, upsample to 640."""
    S = img.shape[1] // zoom
    x0 = int(np.clip(round(u - S / 2), 0, img.shape[1] - S))
    y0 = int(np.clip(round(foot_row - 0.6 * S), 0, img.shape[0] - S))
    tile = cv2.resize(img[y0:y0 + S, x0:x0 + S], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return tile, (x0, y0)


def _init():
    global _DET
    cv2.setNumThreads(1)
    from .detector import YoloxDetector
    _DET = YoloxDetector(engine=cv2.dnn.ENGINE_NEW, conf=CONF)


def _score(img, box, cls):
    """(best score of `cls` overlapping box, best any-class (cls, score) overlapping box, all dets)."""
    t = time.perf_counter()
    dets = _DET.detect(img)
    ms = (time.perf_counter() - t) * 1000
    best, best_any = 0.0, ("", 0.0)
    for d in dets:
        if box is not None and iou(d.box, box) > IOU_MIN:
            if d.cls == cls:
                best = max(best, d.score)
            if d.score > best_any[1]:
                best_any = (d.cls, d.score)
    return best, best_any, dets, ms


def run_background(task):
    """All jobs for one background frame. Returns a list of records."""
    bg, kinds, seed, zoom, distances = task
    frame = C.load_frame(bg["traj"], bg["frame"])
    recs = []
    rng = np.random.default_rng(seed)

    def one(kind, cid, z, bearing, target_cls):
        for beta in BETAS:
            t = time.perf_counter()
            r = C.render(frame, [(cid, z, bearing)], beta=beta, stereo=False)
            rms = (time.perf_counter() - t) * 1000
            vis = r["visible"][0]
            box = mask_box(r["masks"][0])
            img = r["left"]
            if zoom > 1:   # "look closer": upsample the tile around the bearing / ground-contact row
                img, (x0, y0) = zoom_tile(img, C.bearing_to_col(bearing), r["placements"][0]["foot_row"], zoom)
                if box is not None:
                    box = ((box[0] - x0) * zoom, (box[1] - y0) * zoom, box[2] * zoom, box[3] * zoom)
            rec = {"kind": kind, "bg": frame.name, "cutout": cid, "z": z, "bearing": bearing, "beta": beta,
                   "visible": round(vis, 3), "foot": r["placements"][0]["foot_method"],
                   "height_px": round(r["placements"][0]["height_px"], 1), "render_ms": round(rms, 1)}
            if vis < VIS_MIN and kind != "decoy":
                rec.update(score=None)          # occluded: not a detector trial
                recs.append(rec)
                continue
            s, s_any, dets, ms = _score(img, box, target_cls)
            rec.update(score=round(s, 4), any_cls=s_any[0], any_score=round(s_any[1], 4), det_ms=round(ms, 1),
                       bear_anywhere=round(max([d.score for d in dets if d.cls == "bear"], default=0.0), 4))
            recs.append(rec)

    if "bear" in kinds:
        for z in distances:
            for bearing in pick_bearings(frame, BEARS[0], z, k=2 if zoom == 1 else 1):
                for cid in BEARS:
                    one("bear", cid, z, bearing, "bear")
    if "person" in kinds:
        cid = PERSONS[seed % len(PERSONS)]
        for z in distances:
            one("person", cid, z, pick_bearings(frame, cid, z, k=1)[0], "person")
    if "decoy" in kinds:
        for cid in DECOY_IDS:
            z = int(rng.choice(DECOY_Z))
            one("decoy", cid, z, pick_bearings(frame, cid, z, k=1)[0], "bear")
    if "clean" in kinds:
        for beta in BETAS:
            img = C.fog(frame.left, np.minimum(frame.depth, C.SKY_CLIP), beta)
            if zoom > 1:   # one zoom tile per clean frame, on the horizon band in a random direction
                img, _ = zoom_tile(img, C.bearing_to_col(float(rng.choice(BEARING_CANDIDATES))),
                                   C.CY + C.F * C.CAM_HEIGHT / 20, zoom)
            t = time.perf_counter()
            dets = _DET.detect(img)
            ms = (time.perf_counter() - t) * 1000
            recs.append({"kind": "clean", "bg": frame.name, "beta": beta, "det_ms": round(ms, 1),
                         "bear_scores": [round(d.score, 4) for d in dets if d.cls == "bear"],
                         "top": [(d.cls, round(d.score, 3)) for d in dets[:5]]})
    return recs


# ---------------------------------------------------------------- analysis
def _wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def fit_surrogate(recs, thr):
    """Logistic p(d, beta) = sigmoid(a + b ln d + c beta d) by maximum likelihood (binary trials)."""
    from scipy.optimize import minimize
    trials = [(r["z"], r["beta"], float(r["score"] >= thr)) for r in recs]
    z, beta, y = (np.array(v, float) for v in zip(*trials))
    X = np.stack([np.ones_like(z), np.log(z), beta * z], 1)

    def nll(w):
        s = X @ w
        return float(np.sum(np.logaddexp(0, s) - y * s))

    def grad(w):
        p = 1 / (1 + np.exp(-(X @ w)))
        return X.T @ (p - y)

    w = minimize(nll, np.zeros(3), jac=grad, method="BFGS").x
    p = 1 / (1 + np.exp(-(X @ w)))
    null = -np.sum(y * np.log(y.mean() + 1e-12) + (1 - y) * np.log(1 - y.mean() + 1e-12))
    cells = {}
    for zz in DISTANCES:
        for bb in BETAS:
            m = (z == zz) & (beta == bb)
            if m.any():
                cells[f"{bb}|{zz}"] = (float(y[m].mean()), float(p[m].mean()), int(m.sum()))
    errs = [abs(o - q) for o, q, _ in cells.values()]
    return {"form": "p = 1 / (1 + exp(-(a + b*ln(d_m) + c*beta*d_m)))", "a": round(float(w[0]), 4),
            "b": round(float(w[1]), 4), "c": round(float(w[2]), 4), "threshold": thr, "n_trials": int(len(y)),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "mcfadden_r2": round(float(1 - nll(w) / null), 4),
            "cell_mae": round(float(np.mean(errs)), 4), "cell_max_abs_err": round(float(np.max(errs)), 4),
            "cells_observed_vs_fit": {k: [round(o, 3), round(q, 3), n] for k, (o, q, n) in cells.items()}}


def summarise(recs, n_frames, workers, wall_s, bgs_used):
    bear = [r for r in recs if r["kind"] == "bear"]
    trials = [r for r in bear if r["score"] is not None]
    out = {"distances_m": list(DISTANCES), "betas": list(BETAS),
           "visibility_m": {str(b): (None if b == 0 else round(3.912 / b, 1)) for b in BETAS},
           "p_detect": {}, "p_detect_ci95": {}, "n_trials": {}, "conf_quantiles": {},
           "line_of_sight_rate": {}}
    for thr in THRESHOLDS:
        out["p_detect"][str(thr)], out["p_detect_ci95"][str(thr)] = {}, {}
        for b in BETAS:
            pd, ci = {}, {}
            for z in DISTANCES:
                s = [r["score"] for r in trials if r["beta"] == b and r["z"] == z]
                k = sum(v >= thr for v in s)
                pd[str(z)] = round(k / len(s), 4) if s else None
                ci[str(z)] = _wilson(k, len(s))
            out["p_detect"][str(thr)][str(b)], out["p_detect_ci95"][str(thr)][str(b)] = pd, ci
    # conf_quantiles: score distribution of *detections* (best bear score >= 0.25) per cell; cells with < 3
    # detections fall back to the pooled detections at that beta (then overall) and are flagged in conf_quantiles_n.
    # conf_quantiles_all: unconditional best-bear-score quantiles (0 = nothing matched).
    out["conf_quantiles_all"], out["conf_quantiles_n"] = {}, {}
    det_all = [r["score"] for r in trials if r["score"] >= 0.25] or [0.25]
    for b in BETAS:
        out["conf_quantiles"][str(b)], out["n_trials"][str(b)] = {}, {}
        out["conf_quantiles_all"][str(b)], out["conf_quantiles_n"][str(b)] = {}, {}
        det_b = [r["score"] for r in trials if r["beta"] == b and r["score"] >= 0.25]
        for z in DISTANCES:
            s = [r["score"] for r in trials if r["beta"] == b and r["z"] == z]
            d = [v for v in s if v >= 0.25]
            out["n_trials"][str(b)][str(z)] = len(s)
            out["conf_quantiles_n"][str(b)][str(z)] = len(d)
            src = d if len(d) >= 3 else (det_b if len(det_b) >= 3 else det_all)
            out["conf_quantiles"][str(b)][str(z)] = [round(float(q), 4) for q in np.quantile(src, [0.1, 0.5, 0.9])]
            out["conf_quantiles_all"][str(b)][str(z)] = [round(float(q), 4) for q in
                                                          np.quantile(s, [0.1, 0.5, 0.9])] if s else None
    for z in DISTANCES:
        rs = [r for r in bear if r["z"] == z and r["beta"] == 0]
        out["line_of_sight_rate"][str(z)] = round(sum(r["score"] is not None for r in rs) / max(len(rs), 1), 3)
    out["p_detect_by_cutout"] = {cid: {str(b): {str(z): round(float(np.mean(
        [r["score"] >= 0.25 for r in trials if r["cutout"] == cid and r["beta"] == b and r["z"] == z] or [np.nan])), 3)
        for z in DISTANCES} for b in BETAS} for cid in BEARS}
    # false 'bear' per clean frame
    clean = [r for r in recs if r["kind"] == "clean"]
    out["false_bear_per_frame_by_threshold"] = {str(thr): {str(b): round(float(np.mean(
        [sum(s >= thr for s in r["bear_scores"]) for r in clean if r["beta"] == b])), 4) for b in BETAS}
        for thr in THRESHOLDS}
    out["clean_frames_with_false_bear"] = {str(thr): {str(b): round(float(np.mean(
        [any(s >= thr for s in r["bear_scores"]) for r in clean if r["beta"] == b])), 4) for b in BETAS}
        for thr in THRESHOLDS}
    out["n_clean_frames_per_beta"] = len([r for r in clean if r["beta"] == 0])
    # decoys
    dec = [r for r in recs if r["kind"] == "decoy" and r["score"] is not None]
    out["decoy_as_bear_by_threshold"] = {str(thr): {str(b): round(float(np.mean(
        [r["score"] >= thr for r in dec if r["beta"] == b])), 4) for b in BETAS} for thr in THRESHOLDS}
    out["false_bear_per_frame"] = out["false_bear_per_frame_by_threshold"][str(THRESHOLDS[0])]
    out["decoy_as_bear"] = out["decoy_as_bear_by_threshold"][str(THRESHOLDS[0])]
    out["false_rates_threshold"] = THRESHOLDS[0]
    out["decoy_as_bear_by_cutout"] = {cid: {str(b): round(float(np.mean(
        [r["score"] >= 0.25 for r in dec if r["beta"] == b and r["cutout"] == cid] or [np.nan])), 3)
        for b in BETAS} for cid in DECOY_IDS}
    out["decoy_top_class_all_betas"] = {}
    for cid in DECOY_IDS:   # what YOLOX called each decoy (top overlapping class, score >= 0.25), all betas
        cnt = {}
        for r in dec:
            if r["cutout"] == cid and r["any_score"] >= 0.25:
                cnt[r["any_cls"]] = cnt.get(r["any_cls"], 0) + 1
        out["decoy_top_class_all_betas"][cid] = dict(sorted(cnt.items(), key=lambda kv: -kv[1]))
    out["decoy_z_m"] = list(DECOY_Z)
    out["n_decoy_trials_per_beta"] = len([r for r in dec if r["beta"] == 0])
    # persons
    per = [r for r in recs if r["kind"] == "person" and r["score"] is not None]
    out["person_p_detect"] = {str(thr): {str(b): {str(z): (round(float(np.mean(s)), 4) if s else None) for z in DISTANCES
                                                  for s in [[r["score"] >= thr for r in per if r["beta"] == b and r["z"] == z]]}
                                         for b in BETAS} for thr in THRESHOLDS}
    out["person_n_trials"] = {str(z): len([r for r in per if r["z"] == z and r["beta"] == 0]) for z in DISTANCES}
    # surrogate
    out["surrogate"] = {str(thr): fit_surrogate(trials, thr) for thr in THRESHOLDS}
    out["surrogate"]["false_cue_rate"] = {
        "definition": "expected false 'bear' cues per clean frame (score >= thr) + P(decoy reported as bear)",
        "per_clean_frame": out["false_bear_per_frame_by_threshold"],
        "per_decoy_sighting": out["decoy_as_bear_by_threshold"]}
    dms = [r["det_ms"] for r in recs if r.get("det_ms")]
    rms = [r["render_ms"] for r in recs if r.get("render_ms")]
    out["n_frames"] = {"detector_calls": len(dms), "bear_trials": len(trials), "bear_placements": len(bear),
                       "backgrounds": len(bgs_used), "clean": len(clean), "decoy": len(dec), "person": len(per)}
    out["timing_ms_per_frame"] = {"yolox_median": round(float(np.median(dms)), 1),
                                  "yolox_p90": round(float(np.quantile(dms, 0.9)), 1),
                                  "composite_median": round(float(np.median(rms)), 1),
                                  "threads_per_worker": 1, "workers": workers, "wall_s": round(wall_s, 1),
                                  "cpu": platform.processor() or platform.machine()}
    man = json.loads((ROOT / "data" / "cutouts" / "manifest.json").read_text())
    out["meta"] = {"opencv": cv2.__version__, "model": "OpenCV Zoo object_detection_yolox_2022nov.onnx (Apache-2.0)",
                   "engine": "cv2.dnn.ENGINE_NEW (=2), CPU", "conf": CONF, "nms": 0.5, "iou_match": IOU_MIN,
                   "min_visible_fraction": VIS_MIN,
                   "placement": "two most-visible bearings in [-38,38] deg per (background, Z); same for all beta",
                   "fog": "I = J exp(-beta d) + A (1 - exp(-beta d)), A = 210, d = GT depth (sky clipped 1000 m)",
                   "camera": "640x640, f = 320 px (90 deg HFOV), stereo baseline 0.25 m",
                   "heights_m": HEIGHT_M,
                   "backgrounds": "TartanGround ForestEnv (Patel et al., arXiv:2505.10696, CC BY 4.0): "
                                  + ", ".join(sorted({b['traj'] for b in bgs_used})),
                   "background_frames": [f"{b['traj']}_{b['frame']:06d}" for b in bgs_used],
                   "cutouts": {k: {"license": v["license"], "author": v["author"], "source": v["source_page"]}
                               for k, v in man.items()}}
    return out


def plot(out, path, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, thr in zip(axes, THRESHOLDS):
        for i, b in enumerate(BETAS):
            pd = out["p_detect"][str(thr)][str(b)]
            ci = out["p_detect_ci95"][str(thr)][str(b)]
            zs = [z for z in DISTANCES if pd[str(z)] is not None]
            ys = [pd[str(z)] for z in zs]
            lo = [ci[str(z)][0] for z in zs]
            hi = [ci[str(z)][1] for z in zs]
            vis = "clear" if b == 0 else f"vis {3.912 / b:.0f} m"
            c = cmap(i / (len(BETAS) - 1) * 0.9)
            ax.plot(zs, ys, "o-", color=c, label=f"β={b} ({vis})")
            ax.fill_between(zs, lo, hi, color=c, alpha=0.12, lw=0)
            s = out["surrogate"][str(thr)]
            zz = np.linspace(5, 60, 100)
            ax.plot(zz, 1 / (1 + np.exp(-(s["a"] + s["b"] * np.log(zz) + s["c"] * b * zz))), "--", color=c, lw=1)
        ax.set_title(f"YOLOX 'bear', score ≥ {thr}  (dashed: logistic surrogate)", fontsize=10)
        ax.set_xlabel("distance Z (m)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("P(detect | ≥50% visible)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Bear detection vs distance and fog — composites on TartanGround ForestEnv, OpenCV 5 DNN" + suffix,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-bg", type=int, default=20, help="backgrounds for the bear/person/decoy grid")
    ap.add_argument("--reanalyse", action="store_true", help="re-summarise the saved records")
    ap.add_argument("--zoom", type=int, default=1, help="'look closer' tile zoom (e.g. 3); writes calibration_zoom*")
    a = ap.parse_args()
    global DISTANCES
    if a.zoom > 1:
        DISTANCES = (10, 15, 20, 30, 40, 60)
    tag = "" if a.zoom == 1 else "_zoom"
    bgs = json.loads((C.TG / "backgrounds.json").read_text())
    # spread the grid backgrounds over all trajectories (every other frame), clean frames use all
    grid = bgs[::max(1, len(bgs) // a.n_bg)][:a.n_bg]
    RESULTS.mkdir(exist_ok=True)
    rec_path = RESULTS / f"calibration{tag}_records.jsonl"
    t0 = time.time()
    if a.reanalyse:
        recs = [json.loads(l) for l in rec_path.read_text().splitlines()]
        # drop trials of retired cut-outs (licence not clear), so results only rest on shippable photos
        n0 = len(recs)
        recs = [r for r in recs if r.get("cutout") not in RETIRED]
        if len(recs) < n0:
            print(f"dropped {n0 - len(recs)} records of retired cut-outs {sorted(RETIRED)}")
    else:
        tasks = []
        for i, bg in enumerate(bgs):
            if a.zoom == 1:
                kinds = ("clean",) + (("bear", "person", "decoy") if bg in grid else ())
            else:
                kinds = ("clean", "bear", "decoy") if bg in grid else ()
            if kinds:
                tasks.append((bg, kinds, i, a.zoom, DISTANCES))
        tasks.sort(key=lambda t: -len(t[1]))     # heavy tasks first
        recs = []
        with mp.get_context("spawn").Pool(min(a.workers, 6), initializer=_init) as pool, \
                open(rec_path, "w") as f:
            for k, rs in enumerate(pool.imap_unordered(run_background, tasks)):
                recs += rs
                for r in rs:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"[{time.time() - t0:6.0f}s] {k + 1}/{len(tasks)} backgrounds, {len(recs)} records", flush=True)
    wall = time.time() - t0
    prev = RESULTS / f"calibration{tag}.json"
    if a.reanalyse and prev.exists():   # keep the wall time of the run that produced the records
        wall = json.loads(prev.read_text()).get("timing_ms_per_frame", {}).get("wall_s", wall)
    out = summarise(recs, len(recs), a.workers, wall, grid)
    out["meta"]["zoom"] = a.zoom
    out["meta"]["bear_cutouts"] = list(BEARS)
    out["meta"]["excluded_cutouts"] = {k: "licence not clear on review; trials dropped" for k in RETIRED}
    if a.zoom > 1:
        out["meta"]["placement"] = "most-visible bearing per (background, Z); detector runs on a (640/zoom)^2 " \
                                   "tile centred on that bearing, upsampled to 640 (agent 'look closer' action)"
        out["meta"]["zoom_caveat"] = ("the tile's vertical position uses the simulator's ground-contact row (oracle). "
                                      "An agent-computable horizon row (cy + f*1.2/Z) is within 85 px of it in 92% of "
                                      "placements (review, 20 grid backgrounds, Z 15-30 m), so zoom P(detect) is an "
                                      "upper bound by up to ~0.08; the bearing itself is also given")
    (RESULTS / f"calibration{tag}.json").write_text(json.dumps(out, indent=1))
    plot(out, RESULTS / f"vision_pdetect{tag}.png", "" if a.zoom == 1 else f" — {a.zoom}x zoom tile")
    print(json.dumps({k: out[k] for k in ("p_detect", "false_bear_per_frame_by_threshold",
                                          "decoy_as_bear_by_threshold", "person_p_detect")}, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cells_observed_vs_fit"}
                      for k, v in out["surrogate"].items() if k != "false_cue_rate"}, indent=1))


if __name__ == "__main__":
    main()

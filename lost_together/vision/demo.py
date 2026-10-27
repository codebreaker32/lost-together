"""Demo gallery for the video: the same forest frame with a bear at fixed range under increasing fog,
a decoy deer frame, and a stereo-depth visualisation. Writes PNGs to data/demo/.

    uv run python -m lost_together.vision.demo [--frame P0009:2414]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import composite as C
from .calibrate import zoom_tile
from .detector import YoloxDetector
from .stereo import StereoDepth, colorize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "demo"


def inset(img, box, scale=3, size=150):
    """Return img with a magnified inset of the region around box (bottom-right corner)."""
    x, y, w, h = map(int, box)
    cx, cy = x + w // 2, y + h // 2
    half = size // (2 * scale)
    x0, y0 = int(np.clip(cx - half, 0, img.shape[1] - 2 * half)), int(np.clip(cy - half, 0, img.shape[0] - 2 * half))
    crop = cv2.resize(img[y0:y0 + 2 * half, x0:x0 + 2 * half], (size, size), interpolation=cv2.INTER_NEAREST)
    out = img.copy()
    cv2.rectangle(out, (x0, y0), (x0 + 2 * half, y0 + 2 * half), (255, 255, 255), 1)
    H, W = out.shape[:2]
    out[H - size - 8:H - 8, W - size - 8:W - 8] = crop
    cv2.rectangle(out, (W - size - 9, H - size - 9), (W - 8, H - 8), (255, 255, 255), 1)
    cv2.putText(out, f"x{scale}", (W - size - 4, H - size + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="P0009:710")
    ap.add_argument("--bear", default="bear_grizzly_rocks")
    ap.add_argument("--bearing", type=float, default=None, help="default: most visible bearing at 30 m")
    a = ap.parse_args()
    cv2.setNumThreads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    traj, idx = a.frame.split(":")
    fr = C.load_frame(traj, int(idx))
    if a.bearing is None:
        from .calibrate import pick_bearings
        a.bearing = float(pick_bearings(fr, a.bear, 30, k=1)[0])
    det = YoloxDetector(conf=0.1)
    log = {}
    rows = []
    for z, zoom in ((10, 1), (30, 1), (20, 3)):
        row = []
        for beta in (0.0, 0.05, 0.1):
            r = C.render(fr, [(a.bear, z, a.bearing)], beta=beta)
            vis = "clear" if beta == 0 else f"visibility {3.912 / beta:.0f} m"
            if zoom == 1:
                dets = [d for d in det.detect(r["left"]) if d.score >= 0.25]
                title = f"bear at {z} m, beta={beta} ({vis})"
                img = inset(C.annotate(r["left"], dets, title=title), r["boxes"][0])
                name = f"bear_{z}m_beta{beta}.png"
            else:   # the agent's "look closer" action: 3x tile around the bearing
                tile, _ = zoom_tile(r["left"], C.bearing_to_col(a.bearing), r["placements"][0]["foot_row"], zoom)
                dets = [d for d in det.detect(tile) if d.score >= 0.25]
                img = C.annotate(tile, dets, title=f"look closer ({zoom}x tile): bear at {z} m, beta={beta}")
                name = f"bear_{z}m_beta{beta}_zoom{zoom}.png"
            cv2.imwrite(str(OUT / name), img)
            log[name] = [(d.cls, round(d.score, 3)) for d in dets]
            row.append(cv2.resize(img, (420, 420), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack(row))
    cv2.imwrite(str(OUT / "fog_strip.png"), np.vstack(rows))
    # decoy deer (and a dog) in the same frame
    for cid, z, b in (("deer_doe_standing", 8, a.bearing), ("deer_doe_grazing", 8, a.bearing),
                      ("dog_shepherd", 6, a.bearing)):
        r = C.render(fr, [(cid, z, b)], beta=0.0)
        dets = [d for d in det.detect(r["left"]) if d.score >= 0.25]
        img = inset(C.annotate(r["left"], dets, title=f"decoy: {cid.split('_')[0]} at {z} m"), r["boxes"][0])
        cv2.imwrite(str(OUT / f"decoy_{cid}.png"), img)
        log[f"decoy_{cid}.png"] = [(d.cls, round(d.score, 3)) for d in dets]
    # bear + hiker together, stereo pair and SGBM depth
    from .calibrate import visibility
    hb = max((b for b in range(-38, 39, 2) if abs(b - a.bearing) >= 12),     # most visible hiker bearing
             key=lambda b: visibility(fr, "person_hiker_path", 8, b))
    r = C.render(fr, [(a.bear, 9, a.bearing), ("person_hiker_path", 8, float(hb))], beta=0.0)
    dets = [d for d in det.detect(r["left"]) if d.score >= 0.25]
    cv2.imwrite(str(OUT / "bear_and_hiker.png"), C.annotate(r["left"], dets, title="bear 9 m, hiker 8 m"))
    log["bear_and_hiker.png"] = [(d.cls, round(d.score, 3)) for d in dets]
    sd = StereoDepth()
    depth, _ = sd.compute(r["left"], r["right"])
    gt = colorize(np.where(r["depth"] < 1000, r["depth"], 0))
    est = colorize(depth)
    lab = lambda im, t: cv2.putText(im.copy(), t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,  # noqa
                                    cv2.LINE_AA)
    strip = np.hstack([lab(r["left"], "left (composite)"), lab(gt, "GT depth"), lab(est, "StereoSGBM depth")])
    cv2.imwrite(str(OUT / "stereo_depth.png"), strip)
    (OUT / "gallery.json").write_text(json.dumps({"frame": fr.name, "detections_score_ge_0.25": log}, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()

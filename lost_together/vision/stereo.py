"""Stereo depth with OpenCV 5 StereoSGBM (PROJECT_BRIEF section 5) and its error vs TartanGround GT under fog.

SGBM settings: MODE_SGBM_3WAY, numDisparities 128, blockSize 5, P1 = 8*cn*b^2, P2 = 32*cn*b^2,
uniquenessRatio 10, speckleWindowSize 100 / speckleRange 2, disp12MaxDiff 1.
Output disparity is int16 x16 (4 fractional bits): depth = f*B / (disp/16); disp/16 <= 1 px is masked invalid.

    uv run python -m lost_together.vision.stereo [--n 20]     # -> results/vision_stereo.json
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import composite as C

ROOT = Path(__file__).resolve().parents[2]
BETAS = (0.0, 0.02, 0.05, 0.1, 0.2)
RANGES = (5, 10, 20)
NOISE_SIGMAS = (0.0, 2.0)   # additive Gaussian sensor noise (grey levels); fog alone is noise-free


def add_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return img
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def make_sgbm(num_disp: int = 128, block: int = 5, cn: int = 1):
    return cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=block,
                                 P1=8 * cn * block * block, P2=32 * cn * block * block, disp12MaxDiff=1,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.StereoSGBM_MODE_SGBM_3WAY)


def disparity_to_depth(disp16: np.ndarray, f: float = C.F, baseline: float = C.BASELINE,
                       min_disp: float = 1.0) -> np.ndarray:
    """int16 fixed-point (x16) disparity -> float32 metres; invalid (disp <= min_disp px) -> 0."""
    d = disp16.astype(np.float32) / 16.0
    out = np.zeros_like(d)
    ok = d > min_disp
    out[ok] = f * baseline / d[ok]
    return out


class StereoDepth:
    def __init__(self, num_disp: int = 128, block: int = 5, gray: bool = True):
        self.gray = gray
        self.sgbm = make_sgbm(num_disp, block, 1 if gray else 3)

    def compute(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.gray:
            left_bgr = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
            right_bgr = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        disp16 = self.sgbm.compute(left_bgr, right_bgr)
        return disparity_to_depth(disp16), disp16


def depth_errors(pred: np.ndarray, gt: np.ndarray, margin: int = 128) -> dict:
    """Errors on pixels left of nothing: skip the left `margin` columns (no right-image overlap)."""
    p, g = pred[:, margin:], gt[:, margin:]
    out = {}
    for r in RANGES:
        m = (g > 0) & (g < r)
        v = m & (p > 0)
        n = int(m.sum())
        if n == 0:
            out[f"<{r}m"] = None
            continue
        err = np.abs(p[v] - g[v])
        rel = err / g[v]
        out[f"<{r}m"] = {"valid_fraction": round(float(v.sum() / n), 4),
                         "median_abs_err_m": round(float(np.median(err)), 4) if v.any() else None,
                         "median_rel_err": round(float(np.median(rel)), 4) if v.any() else None,
                         "bad_rel_gt_10pct": round(float((rel > 0.1).mean()), 4) if v.any() else None,
                         "n_px": n}
    return out


def colorize(depth: np.ndarray, max_d: float = 30.0) -> np.ndarray:
    """Turbo colour map of inverse depth (near = warm), black where invalid."""
    inv = np.where(depth > 0, np.clip(1.5 / np.maximum(depth, 1e-3), 0, 1), 0)
    inv = np.where((depth > 0) & (depth > max_d), 1.5 / max_d, inv)
    vis = cv2.applyColorMap((inv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[depth <= 0] = 0
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    cv2.setNumThreads(1)
    bgs = json.loads((C.TG / "backgrounds.json").read_text())
    sel = bgs[1::max(1, len(bgs) // a.n)][:a.n]       # offset from the calibration grid backgrounds
    sd = StereoDepth()
    rng = np.random.default_rng(0)
    conds = [(b, n) for n in NOISE_SIGMAS for b in BETAS]
    key = lambda b, n: f"beta={b},noise={n:g}"   # noqa: E731
    per = {key(b, n): [] for b, n in conds}
    times = []
    for bg in sel:
        fr = C.load_frame(bg["traj"], bg["frame"])
        gt = np.minimum(fr.depth, C.SKY_CLIP)
        rgt = C.warp_depth_to_right(gt)
        for b, n in conds:
            L = add_noise(C.fog(fr.left, gt, b), n, rng)
            R = add_noise(C.fog(fr.right, rgt, b), n, rng)
            t = time.perf_counter()
            pred, _ = sd.compute(L, R)
            times.append((time.perf_counter() - t) * 1000)
            per[key(b, n)].append(depth_errors(pred, fr.depth))
    summary = {}
    for b, n in conds:
        k = key(b, n)
        summary[k] = {}
        for r in RANGES:
            rows = [e[f"<{r}m"] for e in per[k] if e[f"<{r}m"] is not None]
            agg = lambda k: round(float(np.median([x[k] for x in rows if x[k] is not None])), 4)  # noqa: E731
            summary[k][f"<{r}m"] = {"valid_fraction": agg("valid_fraction"),
                                         "median_abs_err_m": agg("median_abs_err_m"),
                                         "median_rel_err": agg("median_rel_err"),
                                         "bad_rel_gt_10pct": agg("bad_rel_gt_10pct"), "n_frames": len(rows)}
    out = {"summary_median_over_frames": summary, "betas": list(BETAS), "noise_sigmas": list(NOISE_SIGMAS),
           "visibility_m": {str(b): (None if b == 0 else round(3.912 / b, 1)) for b in BETAS},
           "frames": [f"{b['traj']}_{b['frame']:06d}" for b in sel], "per_frame": per,
           "timing_ms_per_frame": {"sgbm_median": round(float(np.median(times)), 1), "threads": 1},
           "meta": {"opencv": cv2.__version__, "api": "cv2.StereoSGBM_create, MODE_SGBM_3WAY",
                    "numDisparities": 128, "blockSize": 5, "P1": "8*cn*b^2", "P2": "32*cn*b^2", "cn": 1,
                    "uniquenessRatio": 10, "speckleWindowSize": 100, "speckleRange": 2,
                    "depth": "f*B/(disp16/16), invalid if disp <= 1 px; evaluation skips the 128 leftmost columns",
                    "noise": "i.i.d. Gaussian, sigma in grey levels, added after fog, independent per view",
                    "fog": "applied to both views (right view with left GT depth forward-warped)",
                    "data": "TartanGround ForestEnv (CC BY 4.0), front stereo pair, 640x640, f=320, B=0.25 m"}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "vision_stereo.json").write_text(json.dumps(out, indent=1))
    for k, v in summary.items():
        print(k, {r: (x["median_abs_err_m"], x["valid_fraction"], x["bad_rel_gt_10pct"]) for r, x in v.items()})
    print(out["timing_ms_per_frame"])


if __name__ == "__main__":
    main()

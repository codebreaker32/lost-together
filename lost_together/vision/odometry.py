"""Frame-to-frame RGB-D visual odometry on TartanGround ForestEnv P0004 (frames 0-120), OpenCV 5 ptcloud.

Estimator per frame pair (i -> i+1), all OpenCV 5:
  1. cv2.Odometry(OdometryType_RGB_DEPTH, or OdometryType_DEPTH = ICP) with explicit OdometrySettings
     (float32 K, maxDepth 25 m, maxTranslation 1.0 m and maxRotation 10 deg for a ~0.45 m / <= 6 deg step at
     10 Hz), fed depth rescaled by DENSE_SCALE = 0.1 (see the finding below).
  2. Cross-check: ORB keypoints + Hamming matching (ratio test), 3D points from frame i's depth,
     cv2.solvePnPRansac against frame i+1 keypoints. The Odometry success flag is not a validity check.
  3. Gate: accept the dense estimate only when PnP has enough inliers (>= 40) and the two agree
     (|dt| < 0.15 m and |dR| < 2 deg); otherwise use the PnP estimate (>= 15 inliers); otherwise a
     constant-velocity prediction. An unverified dense estimate is never used.

Finding (opencv-python 5.0.0.93): with depth in metres, Odometry.compute returns ok=True with a plausible
rotation but a translation of ~1e-4 m for a 0.42 m step, for RGB_DEPTH and DEPTH, whatever maxDepth /
maxTranslation / maxRotation / maxDepthDiff are set to (setMinDepth makes it return false). Feeding depth in
units of 10 m (depth * 0.1, limits scaled to match, translation scaled back) unlocks the translation: on
GT depth, ICP (OdometryType_DEPTH) is then accurate to a few mm on forward steps but fails on turning or
sideways steps (median error 0.20 m over the 120 pairs, < 5 cm on 13%), and RGB_DEPTH is worse (median 0.33 m).
The Rt argument is an output only (no initial guess), so dense refinement of the PnP estimate is not possible.
The gate therefore rejects the dense estimate on most pairs; this is exactly the failure the gate exists for.

Depth sources: GT depth, SGBM stereo depth (stereo.py), and SGBM on fogged frames (beta 0.05, 0.1).
Reports ATE (anchored at the shared start pose, and after rigid alignment) and RPE vs GT poses.

    uv run python -m lost_together.vision.odometry      # -> results/vision_odometry.json
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import composite as C
from .stereo import StereoDepth, add_noise

ROOT = Path(__file__).resolve().parents[2]
POSES = C.TG / "ForestEnv_P0004_meta" / "x" / "pose_lcam_front.txt"
FRAMES = range(0, 121)
MAX_DEPTH = 25.0
DENSE_SCALE = 0.1   # metres -> 10 m units for cv2.Odometry (translation is ~0 at metric scale; see docstring)
# body (x fwd, y right, z down) -> optical (x right, y down, z fwd): columns = optical axes in body coords
R_BODY_OPT = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], np.float64)


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def load_gt_poses(path=POSES, frames=FRAMES) -> np.ndarray:
    """4x4 camera-to-world poses of the left optical frame (world NED)."""
    raw = np.loadtxt(path)
    out = []
    for i in frames:
        x, y, z, qx, qy, qz, qw = raw[i]
        T = np.eye(4)
        T[:3, :3] = quat_to_R(qx, qy, qz, qw) @ R_BODY_OPT
        T[:3, 3] = (x, y, z)
        out.append(T)
    return np.array(out)


def make_odometry(max_depth=MAX_DEPTH, max_trans=1.0, max_rot_deg=10.0, kind="RGB_DEPTH", scale=DENSE_SCALE):
    """cv2.Odometry for depth that will be passed multiplied by `scale` (limits are scaled to match)."""
    s = cv2.OdometrySettings()
    s.setCameraMatrix(C.K.astype(np.float32))
    s.setMaxDepth(max_depth * scale)
    s.setMaxTranslation(max_trans * scale)
    s.setMaxRotation(max_rot_deg)
    return cv2.Odometry(getattr(cv2, f"OdometryType_{kind}"), s, cv2.OdometryAlgoType_COMMON)


def dense_estimate(odo, d0, g0, d1, g1, kind="RGB_DEPTH", scale=DENSE_SCALE):
    """Dense motion mapping frame-i points into frame-(i+1) coordinates (metres), or None if compute() fails."""
    args = (d0 * scale, g0, d1 * scale, g1) if kind == "RGB_DEPTH" else (d0 * scale, d1 * scale)
    ok, Rt = odo.compute(*args)
    if not ok:
        return None
    T = np.array(Rt, np.float64)
    T[:3, 3] /= scale
    return T


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


class PnPChecker:
    """ORB + depth + solvePnPRansac: motion that maps frame-i camera points into frame-(i+1) coordinates."""

    def __init__(self, nfeatures=2000, reproj=1.5):
        self.reproj = reproj
        self.orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def features(self, gray):
        return self.orb.detectAndCompute(gray, None)

    def estimate(self, f0, depth0, f1):
        (k0, d0), (k1, d1) = f0, f1
        if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
            return None, 0
        pairs = self.bf.knnMatch(d0, d1, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < 0.8 * rest[0].distance]
        obj, img = [], []
        for m in good:
            u, v = k0[m.queryIdx].pt
            z = depth0[int(round(v)), int(round(u))] if 0 <= int(round(v)) < depth0.shape[0] and \
                0 <= int(round(u)) < depth0.shape[1] else 0
            if 0.3 < z < MAX_DEPTH:
                obj.append(((u - C.CX) * z / C.F, (v - C.CY) * z / C.F, z))
                img.append(k1[m.trainIdx].pt)
        if len(obj) < 8:
            return None, 0
        ok, rvec, tvec, inl = cv2.solvePnPRansac(np.array(obj, np.float64), np.array(img, np.float64), C.K, None,
                                                 iterationsCount=300, reprojectionError=self.reproj,
                                                 confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inl is None or len(inl) < 6:
            return None, 0
        i = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(np.array(obj, np.float64)[i], np.array(img, np.float64)[i], C.K, None,
                                          rvec, tvec)
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        return T, int(len(inl))


def run(frames, depths, mode="gated", min_inliers=40, weak_inliers=15, agree_t=0.08, agree_deg=1.5,
        max_step_t=1.0, max_step_deg=10.0, dense_kind="RGB_DEPTH", dense_cache=None):
    """frames: list of BGR; depths: list of float32 depth (0 invalid). Returns (poses 4x4 cam-to-start, log).

    dense_kind: "RGB_DEPTH" or "DEPTH" (ICP). dense_cache: optional dict reused across modes on the same inputs.
    agree_t / agree_deg: dense is accepted only within ~2x the PnP median step error (4 cm on GT depth)."""
    odo, pnp = make_odometry(kind=dense_kind), PnPChecker()
    cache = {} if dense_cache is None else dense_cache
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    feats = [pnp.features(g) for g in grays] if mode != "dense" else None
    poses, log = [np.eye(4)], []
    last = np.eye(4)
    for i in range(len(frames) - 1):
        t0 = time.perf_counter()
        d0 = np.where(depths[i] < MAX_DEPTH, depths[i], 0).astype(np.float32)
        d1 = np.where(depths[i + 1] < MAX_DEPTH, depths[i + 1], 0).astype(np.float32)
        Td = None
        if mode != "pnp":
            key = (dense_kind, i)
            if key not in cache:
                ts = time.perf_counter()
                Tc = dense_estimate(odo, d0, grays[i], d1, grays[i + 1], dense_kind)
                cache[key] = (Tc, (time.perf_counter() - ts) * 1000)
            Td, dense_ms = cache[key]
        else:
            dense_ms = 0.0
        t1 = time.perf_counter()
        Tp, ninl = pnp.estimate(feats[i], d0, feats[i + 1]) if mode != "dense" else (None, 0)
        t2 = time.perf_counter()
        # all estimates map frame-i points into frame-(i+1) coordinates: p_{i+1} = T p_i
        within = lambda T: T is not None and np.linalg.norm(T[:3, 3]) < max_step_t and \
            rot_deg(T[:3, :3]) < max_step_deg   # noqa: E731
        if mode == "dense":
            T, src = (Td, "dense") if within(Td) else (last, "const_vel")
        elif mode == "pnp":
            T, src = (Tp, "pnp") if (ninl >= weak_inliers and within(Tp)) else (last, "const_vel")
        else:
            if Tp is not None and ninl >= min_inliers and within(Tp):
                if within(Td):
                    D = np.linalg.inv(Tp) @ Td
                    agree = np.linalg.norm(D[:3, 3]) < agree_t and rot_deg(D[:3, :3]) < agree_deg
                else:
                    agree = False
                T, src = (Td, "dense") if agree else (Tp, "pnp")
            elif Tp is not None and ninl >= weak_inliers and within(Tp):
                T, src = Tp, "pnp_weak"
            else:   # never trust an unverified dense estimate (ok=True is returned even when it collapses)
                T, src = last, "const_vel"
        last = T
        poses.append(poses[-1] @ np.linalg.inv(T))     # cam_{i+1} -> start
        log.append({"i": i, "src": src, "inliers": ninl,
                    "dense_t_m": None if Td is None else round(float(np.linalg.norm(Td[:3, 3])), 4),
                    "dense_rot_deg": None if Td is None else round(rot_deg(Td[:3, :3]), 3), "dense_ms": round(dense_ms, 1),
                    "pnp_ms": round((t2 - t1) * 1000, 1)})
    return np.array(poses), log


def umeyama_rigid(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mu_d - R @ mu_s


def metrics(est, gt):
    """est, gt: 4x4 cam-to-start poses (gt re-anchored at its first pose)."""
    pe, pg = est[:, :3, 3], gt[:, :3, 3]
    ate_anchor = np.linalg.norm(pe - pg, axis=1)
    R, t = umeyama_rigid(pe, pg)
    ate_aligned = np.linalg.norm(pe @ R.T + t - pg, axis=1)
    rt, rr = [], []
    for i in range(len(est) - 1):
        de = np.linalg.inv(est[i]) @ est[i + 1]
        dg = np.linalg.inv(gt[i]) @ gt[i + 1]
        E = np.linalg.inv(dg) @ de
        rt.append(np.linalg.norm(E[:3, 3]))
        rr.append(rot_deg(E[:3, :3]))
    path = float(np.sum(np.linalg.norm(np.diff(pg, axis=0), axis=1)))
    return {"ate_rmse_anchored_m": round(float(np.sqrt(np.mean(ate_anchor ** 2))), 4),
            "ate_rmse_aligned_m": round(float(np.sqrt(np.mean(ate_aligned ** 2))), 4),
            "final_drift_m": round(float(ate_anchor[-1]), 4),
            "final_drift_pct_of_path": round(float(100 * ate_anchor[-1] / path), 3),
            "rpe_trans_rmse_m": round(float(np.sqrt(np.mean(np.square(rt)))), 4),
            "rpe_trans_median_m": round(float(np.median(rt)), 4),
            "rpe_rot_rmse_deg": round(float(np.sqrt(np.mean(np.square(rr)))), 4),
            "path_length_m": round(path, 2)}


MODES_DOC = {"dense": ("dense", "RGB_DEPTH", "RGB-D Odometry only (const-velocity if outside limits)"),
             "dense_icp": ("dense", "DEPTH", "ICP Odometry only (OdometryType_DEPTH)"),
             "pnp": ("pnp", "RGB_DEPTH", "ORB + depth + solvePnPRansac only"),
             "gated": ("gated", "RGB_DEPTH", "RGB-D Odometry gated by PnP (fallback PnP)"),
             "gated_icp": ("gated", "DEPTH", "ICP Odometry gated by PnP (fallback PnP)")}
MODES = {k: v[:2] for k, v in MODES_DOC.items()}


def plot_trajectories(path=ROOT / "results" / "vision_odometry.png"):
    """Top-down (x-z of the start camera frame) GT vs gated VO trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = ROOT / "results"
    gt = np.load(res / "vision_odometry_traj_gt.npy")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(gt[:, 0], gt[:, 2], "k-", lw=2.5, label="ground truth")
    for f in sorted(res.glob("vision_odometry_traj_*.npy")):
        name = f.stem.replace("vision_odometry_traj_", "")
        if name != "gt":
            t = np.load(f)
            ax.plot(t[:, 0], t[:, 2], lw=1.3, label=f"VO ({name.replace('_', ' ')})")
    ax.set_aspect("equal")
    ax.set_xlabel("x right (m)")
    ax.set_ylabel("z forward (m)")
    ax.set_title("ORB+depth PnP odometry gated against OpenCV 5 RGB-D Odometry\n"
                 "TartanGround ForestEnv P0004, 120 frames", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    cv2.setNumThreads(2)
    gt_w = load_gt_poses()
    gt = np.array([np.linalg.inv(gt_w[0]) @ T for T in gt_w])
    frames = [C.load_frame("P0004", i) for i in FRAMES]
    sd = StereoDepth()
    rng = np.random.default_rng(0)
    out = {"gt_step_m": round(float(np.mean(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1))), 4),
           "gt_step_rot_deg_max": round(max(rot_deg((np.linalg.inv(gt[i]) @ gt[i + 1])[:3, :3])
                                             for i in range(len(gt) - 1)), 3),
           "runs": {}}
    conds = {"gt_depth": (0.0, "gt", 0.0), "sgbm": (0.0, "sgbm", 0.0), "sgbm_fog0.05": (0.05, "sgbm", 0.0),
             "sgbm_fog0.05_noise2": (0.05, "sgbm", 2.0), "sgbm_fog0.1": (0.1, "sgbm", 0.0)}
    for name, (beta, src, noise) in conds.items():
        imgs, depths, sg_ms = [], [], []
        for fr in frames:
            gtd = np.minimum(fr.depth, C.SKY_CLIP)
            L = add_noise(C.fog(fr.left, gtd, beta), noise, rng)
            if src == "gt":
                d = fr.depth.astype(np.float32)
            else:
                R = add_noise(C.fog(fr.right, C.warp_depth_to_right(gtd), beta), noise, rng)
                t = time.perf_counter()
                d, _ = sd.compute(L, R)
                sg_ms.append((time.perf_counter() - t) * 1000)
            imgs.append(L)
            depths.append(d)
        cache = {}
        for label, (mode, kind) in MODES.items():
            t = time.perf_counter()
            est, log = run(imgs, depths, mode, dense_kind=kind, dense_cache=cache)
            wall = time.perf_counter() - t
            m = metrics(est, gt)
            srcs = {}
            for r in log:
                srcs[r["src"]] = srcs.get(r["src"], 0) + 1
            dt = [r["dense_t_m"] for r in log if r["dense_t_m"] is not None]
            m.update(dense_ok_rate=round(len(dt) / len(log), 3),
                     dense_translation_median_m=round(float(np.median(dt)), 4) if dt else None,
                     sources=srcs, median_inliers=float(np.median([r["inliers"] for r in log])),
                     ms_per_pair_excl_cached_dense=round(1000 * wall / len(log), 1),
                     dense_ms_median=round(float(np.median([r["dense_ms"] for r in log])), 1),
                     sgbm_ms_median=round(float(np.median(sg_ms)), 1) if sg_ms else None)
            dense_err = [float(np.linalg.norm((np.linalg.inv(np.linalg.inv(gt[i + 1]) @ gt[i]) @ cache[(kind, i)][0])[:3, 3]))
                         for i in range(len(log)) if mode != "pnp" and cache.get((kind, i), (None,))[0] is not None]
            m.update(dense_step_err_median_m=round(float(np.median(dense_err)), 4) if dense_err else None,
                     dense_step_err_lt5cm_frac=round(float(np.mean(np.array(dense_err) < 0.05)), 3) if dense_err else None)
            out["runs"][f"{name}/{label}"] = m
            print(f"{name:22s} {label:9s} ATE(anch) {m['ate_rmse_anchored_m']:.3f}  ATE(al) "
                  f"{m['ate_rmse_aligned_m']:.3f}  RPE {m['rpe_trans_rmse_m']:.3f} m {m['rpe_rot_rmse_deg']:.2f} deg "
                  f"drift {m['final_drift_pct_of_path']:.2f}%  {srcs}", flush=True)
            if label == "gated":
                np.save(ROOT / "results" / f"vision_odometry_traj_{name}.npy", est[:, :3, 3])
    np.save(ROOT / "results" / "vision_odometry_traj_gt.npy", gt[:, :3, 3])
    out["meta"] = {"opencv": cv2.__version__, "sequence": "TartanGround ForestEnv omni/P0004 front, frames 0-120 "
                   "(CC BY 4.0, Patel et al., arXiv:2505.10696)",
                   "odometry": "cv2.Odometry(OdometryType_RGB_DEPTH or OdometryType_DEPTH, OdometryAlgoType_COMMON); "
                               "maxDepth 25 m, maxTranslation 1.0 m, maxRotation 10 deg; depth and limits passed in "
                               f"units of {1 / DENSE_SCALE:g} m (at metric scale compute() returns ~0 translation; "
                               "setMinDepth makes it return false)",
                   "gate": "ORB(2000)+ratio 0.8, depth back-projection, solvePnPRansac(reproj 1.5 px)+LM refine; "
                           "accept dense if PnP inliers >= 40 and |dt| < 0.08 m, |dR| < 1.5 deg vs PnP, else PnP "
                           "(>= 15 inliers), else constant velocity",
                   "modes": {k: v[2] for k, v in MODES_DOC.items()},
                   "dense_step_err": "per-pair translation error of the dense estimate alone vs GT (m)",
                   "ate": "anchored = no alignment, both start at the shared start pose; aligned = rigid Umeyama",
                   "noise": "i.i.d. Gaussian sigma in grey levels after fog"}
    (ROOT / "results" / "vision_odometry.json").write_text(json.dumps(out, indent=1))
    plot_trajectories()


if __name__ == "__main__":
    main()

"""Composite cut-outs into TartanGround stereo frames at a metric distance, with GT-depth occlusion and fog.

Geometry (pinhole, f = 320 px, c = 320, baseline B = 0.25 m, right camera at +x of the left optical frame):
  * an object at planar depth Z (m) and bearing theta sits at column u = cx + f * tan(theta);
  * its pixel height is f * H / Z (H = real height of the cut-out class);
  * its ground-contact (foot) row is where the GT depth along that bearing is ~Z on a ground-like
    segmentation class, falling back to the geometric horizon row cy + f * h_cam / Z;
  * a pixel of the object is visible only where the scene depth is farther than Z (per-pixel occlusion);
  * in the right image it is shifted left by the disparity f * B / Z, and occluded against the left depth
    forward-warped into the right view (z-buffered).
Fog (Koschmieder / atmospheric scattering) is applied to the whole composite:
  I = J * exp(-beta d) + A * (1 - exp(-beta d)), d = scene depth (sky clipped to 1000 m), Z on object pixels.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .cutouts import HEIGHT_M, cls_of, load

ROOT = Path(__file__).resolve().parents[2]
TG = ROOT / "data" / "tartanground"
F, CX, CY, BASELINE = 320.0, 320.0, 320.0, 0.25
K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1]], np.float64)
SKY_CLIP = 1000.0
AIRLIGHT = 210.0
CAM_HEIGHT = 1.2          # fallback camera height above ground (m); TartanGround robot_height ~1.19
GROUND_IDS = (64, 59, 72, 36)   # ground, groundembankment, grass, groundplant (ForestEnv seg ids)
SEG = {"bush": 6, "foliage": 191, "grass": 72, "ground": 64, "groundembankment": 59, "groundplant": 36,
       "rock": 195, "sky": 27, "stump": 180, "treetrunk": 199}


@dataclass
class Frame:
    left: np.ndarray                 # BGR uint8
    right: np.ndarray                # BGR uint8
    depth: np.ndarray                # float32 metres, planar z, left camera
    seg: np.ndarray | None = None    # uint8 class ids, left camera
    name: str = ""
    right_depth: np.ndarray | None = field(default=None, repr=False)


def read_depth(path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(d.view("<f4").squeeze(-1))


def load_frame(traj: str, idx: int, root: Path = TG) -> Frame:
    base = root / f"ForestEnv_{traj}"
    left = cv2.imread(str(base / "image_lcam_front" / f"{idx:06d}_lcam_front.png"))
    right = cv2.imread(str(base / "image_rcam_front" / f"{idx:06d}_rcam_front.png"))
    depth = read_depth(base / "depth_lcam_front" / f"{idx:06d}_lcam_front_depth.png")
    sp = base / "seg_lcam_front" / f"{idx:06d}_lcam_front_seg.png"
    seg = cv2.imread(str(sp), cv2.IMREAD_UNCHANGED) if sp.exists() else None
    if left is None or right is None:
        raise FileNotFoundError(f"{traj} frame {idx}")
    return Frame(left, right, depth, seg, f"{traj}_{idx:06d}")


def fog(img: np.ndarray, depth: np.ndarray, beta: float, airlight: float = AIRLIGHT) -> np.ndarray:
    """Atmospheric scattering model; depth in metres (clipped at SKY_CLIP)."""
    if beta <= 0:
        return img.copy()
    t = np.exp(-beta * np.minimum(depth, SKY_CLIP)).astype(np.float32)[..., None]
    out = img.astype(np.float32) * t + airlight * (1.0 - t)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def warp_depth_to_right(depth: np.ndarray, f: float = F, baseline: float = BASELINE) -> np.ndarray:
    """Forward-warp left planar depth into the right camera (rectified, right camera at +x), z-buffered."""
    h, w = depth.shape
    d = np.minimum(depth, SKY_CLIP).astype(np.float32)
    ur = np.rint(np.arange(w)[None, :] - f * baseline / d).astype(np.int64)
    rows = np.broadcast_to(np.arange(h)[:, None], (h, w))
    ok = (ur >= 0) & (ur < w)
    flat = rows[ok] * w + ur[ok]
    out = np.full(h * w, np.inf, np.float32)
    np.minimum.at(out, flat, d[ok])
    out = out.reshape(h, w)
    holes = ~np.isfinite(out)
    if holes.any():   # disocclusions see the background: fill from the farther horizontal neighbour
        filled = np.where(holes, 0, out)
        for _ in range(8):
            grown = cv2.dilate(filled, np.ones((1, 3), np.uint8))
            filled = np.where(holes & (filled == 0), grown, filled)
        # wider holes (mainly the right-border band the left camera never sees, up to f*B/z_min px wide):
        # extend the nearest valid depth along the row instead of treating them as sky (which fogs them white)
        miss = filled == 0
        if miss.any() and not miss.all():
            idx = np.where(~miss, np.arange(w)[None, :], 0)
            np.maximum.accumulate(idx, axis=1, out=idx)                    # last valid column to the left
            fwd = np.take_along_axis(filled, idx, 1)
            ridx = np.where(~miss, np.arange(w)[None, :], w - 1)
            ridx = np.minimum.accumulate(ridx[:, ::-1], axis=1)[:, ::-1]   # next valid column to the right
            bwd = np.take_along_axis(filled, ridx, 1)
            filled = np.where(miss, np.where(fwd > 0, fwd, bwd), filled)
        out = np.where(filled == 0, SKY_CLIP, filled)
    return out


def bearing_to_col(bearing_deg: float) -> float:
    return CX + F * np.tan(np.deg2rad(bearing_deg))


def foot_row(frame: Frame, z: float, u: float, half_w: float, tol: float = 0.08) -> tuple[int, str]:
    """Row where the object touches the ground at depth z along column u (and how it was found)."""
    h, w = frame.depth.shape
    geometric = int(round(CY + F * CAM_HEIGHT / z))
    if frame.seg is not None:
        ground = np.isin(frame.seg, GROUND_IDS)
        for hw, how in ((max(half_w, 6.0), "seg+depth"), (max(3 * half_w, 40.0), "seg+depth(wide)")):
            c0, c1 = int(max(0, u - hw)), int(min(w, u + hw + 1))
            d = frame.depth[:, c0:c1]
            m = ground[:, c0:c1] & (np.abs(d - z) < max(tol * z, 0.3))
            if m.sum() >= 3:
                rows = np.nonzero(m)[0]
                return int(np.median(rows)), how
    return min(max(geometric, 0), h - 1), "horizon"


def _harmonize(obj_bgr: np.ndarray, alpha: np.ndarray, bg: np.ndarray, strength: float) -> np.ndarray:
    """Partial luminance match to the local background (photos are sunlit; forest frames often are not)."""
    if strength <= 0 or alpha.sum() < 1:
        return obj_bgr
    a = alpha > 0.5
    lo = float(cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2GRAY)[a].mean() + 1)
    lb = float(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).mean() + 1)
    gain = np.clip((lb / lo) ** strength, 0.7, 1.4)
    return np.clip(obj_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _paste(img, depth_out, obj, alpha, x0, y0, z, scene_depth, margin):
    """Alpha-blend obj into img at (x0, y0) where scene_depth > z (occlusion). Returns visible mask (full)."""
    h, w = img.shape[:2]
    oh, ow = alpha.shape
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + ow, w), min(y0 + oh, h)
    vis = np.zeros((h, w), np.float32)
    if xa >= xb or ya >= yb:
        return vis
    a = alpha[ya - y0:yb - y0, xa - x0:xb - x0]
    o = obj[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
    unocc = (scene_depth[ya:yb, xa:xb] > z - margin).astype(np.float32)
    a = a * unocc
    roi = img[ya:yb, xa:xb].astype(np.float32)
    img[ya:yb, xa:xb] = np.clip(a[..., None] * o + (1 - a[..., None]) * roi + 0.5, 0, 255).astype(np.uint8)
    dz = depth_out[ya:yb, xa:xb]
    depth_out[ya:yb, xa:xb] = np.where(a > 0.5, np.minimum(dz, z), dz)
    vis[ya:yb, xa:xb] = a
    return vis


def render(background: Frame, objects, beta: float = 0.0, airlight: float = AIRLIGHT,
           harmonize: float = 0.35, margin: float = 0.15, stereo: bool = True) -> dict:
    """Composite objects = [(cutout_id, Z_m, bearing_deg), ...] into a stereo frame, then add fog.

    stereo=False skips the right image (returned as None).
    Returns {left, right, depth (left, with objects), right_depth, masks: [alpha (left) float32 0..1],
    masks_right, boxes: [xywh of the full (unoccluded) object extent, left], visible: [fraction visible],
    placements: [dict]}.
    """
    left, right = background.left.copy(), background.right.copy()
    depth = np.minimum(background.depth, SKY_CLIP).astype(np.float32)
    if not stereo:
        rdepth_scene = depth
    elif background.right_depth is not None:
        rdepth_scene = background.right_depth
    else:
        rdepth_scene = warp_depth_to_right(depth)
    rdepth_scene = np.minimum(rdepth_scene, SKY_CLIP).astype(np.float32)
    depth_out, rdepth_out = depth.copy(), rdepth_scene.copy()
    # paint far objects first so nearer ones overwrite them
    order = sorted(range(len(objects)), key=lambda i: -objects[i][1])
    masks, masks_r, boxes, visible, placements = [None] * len(objects), [None] * len(objects), \
        [None] * len(objects), [None] * len(objects), [None] * len(objects)
    for i in order:
        cid, z, bearing = objects[i]
        bgra = load(cid)
        hpx = F * HEIGHT_M[cls_of(cid)] / z
        s = hpx / bgra.shape[0]
        ow, oh = max(1, int(round(bgra.shape[1] * s))), max(1, int(round(hpx)))
        small = cv2.resize(bgra, (ow, oh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        obj, alpha = small[..., :3], small[..., 3].astype(np.float32) / 255.0
        u = bearing_to_col(bearing)
        vf, how = foot_row(background, z, u, ow / 2)
        x0, y0 = int(round(u - ow / 2)), int(round(vf - oh + 1))
        x0c, y0c, x1c, y1c = max(0, x0 - ow), max(0, y0 - oh), min(left.shape[1], x0 + 2 * ow), \
            min(left.shape[0], y0 + 2 * oh)
        local = background.left[y0c:y1c, x0c:x1c] if (x1c > x0c and y1c > y0c) else background.left
        obj = _harmonize(obj, alpha, local, harmonize)
        masks[i] = _paste(left, depth_out, obj, alpha, x0, y0, z, depth, margin)
        disp = F * BASELINE / z
        if stereo:
            masks_r[i] = _paste(right, rdepth_out, obj, alpha, int(round(x0 - disp)), y0, z, rdepth_scene, margin)
        boxes[i] = (float(x0), float(y0), float(ow), float(oh))
        visible[i] = float(masks[i].sum() / max(alpha.sum(), 1e-6))
        placements[i] = {"cutout": cid, "z": z, "bearing_deg": bearing, "col": u, "foot_row": vf,
                         "foot_method": how, "height_px": hpx, "disparity_px": disp}
    left = fog(left, depth_out, beta, airlight)
    right = fog(right, rdepth_out, beta, airlight) if stereo else None
    return {"left": left, "right": right, "depth": depth_out, "right_depth": rdepth_out, "masks": masks,
            "masks_right": masks_r, "boxes": boxes, "visible": visible, "placements": placements}


COLORS = {"bear": (40, 40, 230), "person": (230, 160, 40), "dog": (40, 200, 230), "horse": (200, 80, 200),
          "cow": (200, 80, 200)}


def annotate(frame: np.ndarray, detections, truth_boxes=(), title: str = "", min_score: float = 0.0) -> np.ndarray:
    """Draw detector boxes (and optional dashed ground-truth boxes) for the demo gallery."""
    out = frame.copy()
    for (x, y, w, h) in truth_boxes:
        x, y, w, h = map(int, (x, y, w, h))
        for k in range(0, max(w, h), 8):     # dashed white rectangle
            if k < w:
                cv2.line(out, (x + k, y), (x + min(k + 4, w), y), (255, 255, 255), 1)
                cv2.line(out, (x + k, y + h), (x + min(k + 4, w), y + h), (255, 255, 255), 1)
            if k < h:
                cv2.line(out, (x, y + k), (x, y + min(k + 4, h)), (255, 255, 255), 1)
                cv2.line(out, (x + w, y + k), (x + w, y + min(k + 4, h)), (255, 255, 255), 1)
    for d in sorted(detections, key=lambda d: d.score):     # highest score drawn last (on top)
        if d.score < min_score:
            continue
        x, y, w, h = map(int, d.box)
        c = COLORS.get(d.cls, (180, 180, 180))
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        label = f"{d.cls} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y - 4 if y - th - 6 > 0 else y + h + th + 4
        cv2.rectangle(out, (x, ty - th - 3), (x + tw + 4, ty + 3), c, -1)
        cv2.putText(out, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if title:
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (30, 30, 30), -1)
        cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out

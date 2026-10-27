"""OpenCV 5 tools the hiker agent can call. All pixel work is OpenCV; the model only picks candidate IDs.

Tools (pure functions on BGR uint8 frames):
  detect_candidates(frame)                -> {"frame": {...visibility...}, "candidates": [{id, cls, score, box, source}]}
  look_closer(frame, candidates, id)      -> (report dict, enhanced zoomed crop)
  compare_frames(prev, cur, candidates, id) -> motion report (Farneback flow inside the box, ego-motion removed)
  query_map(summary)                      -> nearest exit, recent sightings, risk near the hiker

`Toolbox` wraps them with a bounded frame store and per-frame candidate cache; the agent loop and the
MCP server both use it, so there is one tool layer.

Candidates come from three passes, so ambiguous shapes reach the agent instead of being dropped:
  1. YOLOX (OpenCV Zoo, OpenCV 5 DNN engine) at the normal threshold          source="yolox"
  2. the same forward pass kept down to a low threshold (default 0.10)        source="yolox_low"
  3. a classical dark-blob pass (background subtraction + contours)           source="blob", cls="unknown"
"""
from __future__ import annotations

import base64
import math
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import DecisionConfig

# Classes the agent cares about. "bear" is the predator; the rest are natural decoys or other hikers.
ANIMALS = ("bear", "dog", "horse", "cow", "sheep", "elephant", "cat", "zebra", "giraffe")
KEEP_CLASSES = ("person",) + ANIMALS
# How much a candidate of each class counts toward predator evidence (shared by rule policy and stub model).
THREAT_WEIGHT = {"bear": 1.0, "unknown": 0.6, "dog": 0.35, "horse": 0.3, "cow": 0.3, "sheep": 0.2,
                 "elephant": 0.5, "cat": 0.2, "zebra": 0.2, "giraffe": 0.2, "person": 0.0}
CONF_HIGH = 0.25
CONF_LOW = 0.10
MAX_CANDIDATES = 6
MAX_FRAME_SIDE = 4096


def threat_weight(cls: str) -> float:
    return THREAT_WEIGHT.get(cls, 0.2)


# --- detector (lazy singleton; the ONNX model loads once per process) --------------------------------
_DET_LOCK = threading.Lock()
_DETECTOR = None
_DETECTOR_FAILED: str | None = None


def get_detector():
    """YOLOX at CONF_LOW, restricted to KEEP_CLASSES. Returns None (blob pass only) if the model is missing."""
    global _DETECTOR, _DETECTOR_FAILED
    with _DET_LOCK:
        if _DETECTOR is None and _DETECTOR_FAILED is None:
            try:
                from ..vision.detector import YoloxDetector
                _DETECTOR = YoloxDetector(conf=CONF_LOW, classes=KEEP_CLASSES)
            except Exception as e:  # missing model file or engine error: degrade, don't crash
                _DETECTOR_FAILED = f"{type(e).__name__}: {e}"
        return _DETECTOR


def detector_status() -> str:
    get_detector()
    return "yolox" if _DETECTOR is not None else f"blob-only ({_DETECTOR_FAILED})"


# --- frame-level visibility -----------------------------------------------------------------------------
def dark_channel(bgr: np.ndarray, patch: int = 15) -> np.ndarray:
    """He et al. dark channel: per-pixel min over colour, then a min filter (erosion) over a patch."""
    mn = bgr.min(axis=2)
    return cv2.erode(mn, cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))


def atmospheric_light(bgr: np.ndarray, dark: np.ndarray) -> np.ndarray:
    n = max(1, dark.size // 1000)                       # brightest 0.1% of the dark channel
    idx = np.argpartition(dark.reshape(-1), -n)[-n:]
    return bgr.reshape(-1, 3)[idx].astype(np.float32).max(axis=0).clip(1, 255)


def visibility(bgr: np.ndarray) -> dict:
    """Haze from the dark-channel prior (0 clear .. 1 dense) and RMS contrast. A per-frame fog estimate."""
    small = _resize_max(bgr, 320)
    dark = dark_channel(small)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    haze = float(np.clip(np.median(dark) / 255.0, 0, 1))
    return {"haze": round(haze, 3), "contrast": round(float(gray.std() / 128.0), 3),
            "visibility": round(1.0 - haze, 3)}


# --- detection --------------------------------------------------------------------------------------------
def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _clip_box(box, shape) -> list[int]:
    h, w = shape[:2]
    x, y, bw, bh = box
    x0, y0 = int(max(0, math.floor(x))), int(max(0, math.floor(y)))
    x1, y1 = int(min(w, math.ceil(x + bw))), int(min(h, math.ceil(y + bh)))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def dark_blobs(bgr: np.ndarray, max_blobs: int = 4) -> list[dict]:
    """Regions much darker than their surroundings (a dark shape in fog or foliage).

    Background = heavy Gaussian blur; darkness = background - pixel; Otsu-style threshold with a floor;
    morphological open/close; contours filtered by area and by aspect (tall thin shapes are trunks).
    Score is capped at 0.5 so a blob is always "ambiguous" until looked at more closely.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    sigma = max(h, w) / 12.0
    bg = cv2.GaussianBlur(gray, (0, 0), sigma)
    diff = cv2.subtract(bg, gray)                         # positive where darker than surroundings
    thr = max(18.0, float(diff.mean() + 2.5 * diff.std()))
    _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    area_min, area_max = 0.0015 * h * w, 0.25 * h * w
    for c in contours:
        area = cv2.contourArea(c)
        if not area_min <= area <= area_max:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bh > 3.5 * bw or bw > 6 * bh:                 # trunk-like or ground-line-like
            continue
        solidity = area / max(1.0, cv2.contourArea(cv2.convexHull(c)))
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        darkness = float(cv2.mean(diff, mask=m)[0])
        score = 0.5 * float(np.clip(darkness / 80.0, 0, 1)) * (0.6 + 0.4 * solidity)
        out.append({"cls": "unknown", "score": round(score, 3), "box": [x, y, bw, bh], "source": "blob"})
    out.sort(key=lambda d: -d["score"])
    return out[:max_blobs]


def detect_candidates(bgr: np.ndarray, use_yolox: bool = True, conf_high: float = CONF_HIGH,
                      max_candidates: int = MAX_CANDIDATES) -> dict:
    _check_frame(bgr)
    cands: list[dict] = []
    det = get_detector() if use_yolox else None
    if det is not None:
        for d in det.detect(bgr):
            cands.append({"cls": d.cls, "score": round(d.score, 3), "box": _clip_box(d.box, bgr.shape),
                          "source": "yolox" if d.score >= conf_high else "yolox_low"})
    for b in dark_blobs(bgr):
        if all(_iou(b["box"], c["box"]) < 0.3 for c in cands):
            cands.append(b)
    # rank by predator relevance first, then score; people (other hikers) stay listed but last
    cands.sort(key=lambda c: (-threat_weight(c["cls"]) * c["score"], -c["score"]))
    cands = cands[:max_candidates]
    for i, c in enumerate(cands):
        c["id"] = i
    return {"frame": visibility(bgr), "detector": "yolox" if det is not None else "blob-only",
            "candidates": [{"id": c["id"], "cls": c["cls"], "score": c["score"], "box": c["box"],
                            "source": c["source"]} for c in cands]}


# --- look closer --------------------------------------------------------------------------------------------
def defog(bgr: np.ndarray, airlight: np.ndarray | None = None, omega: float = 0.9, t_min: float = 0.2) -> np.ndarray:
    """Single-image dehazing with the dark-channel prior: J = (I - A) / max(t, t_min) + A."""
    img = bgr.astype(np.float32)
    dark = dark_channel(bgr, patch=max(3, (min(bgr.shape[:2]) // 20) | 1))
    A = atmospheric_light(bgr, dark) if airlight is None else airlight.astype(np.float32)
    norm_dark = dark_channel(np.clip(img / A * 255.0, 0, 255).astype(np.uint8), patch=7).astype(np.float32) / 255.0
    t = 1.0 - omega * norm_dark
    t = cv2.GaussianBlur(t, (0, 0), 3)                   # cheap stand-in for the guided-filter refinement
    t = np.maximum(t, t_min)[..., None]
    return np.clip((img - A) / t + A, 0, 255).astype(np.uint8)


def clahe(bgr: np.ndarray, clip: float = 2.5, grid: int = 4) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _find(cands: list[dict], cid: int) -> dict:
    for c in cands:
        if c["id"] == cid:
            return c
    raise ToolError(f"unknown candidate_id {cid}; valid ids: {[c['id'] for c in cands]}")


def look_closer(bgr: np.ndarray, cands: list[dict], cid: int, zoom: float = 2.5,
                use_yolox: bool = True) -> tuple[dict, np.ndarray]:
    """Crop the candidate with context, defog (dark-channel prior), CLAHE, upscale 2-3x, and re-detect."""
    zoom = float(np.clip(zoom, 2.0, 3.0))
    c = _find(cands, cid)
    H, W = bgr.shape[:2]
    x, y, w, h = c["box"]
    side = max(w, h) * 1.6 + 16                          # square crop with context around the box
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    crop = bgr[y0:y1, x0:x1]
    full_dark = dark_channel(_resize_max(bgr, 320))
    A = atmospheric_light(_resize_max(bgr, 320), full_dark)
    enhanced = clahe(defog(crop, A))
    out_side = min(512.0, max(crop.shape[:2]) * zoom)
    scale = out_side / max(crop.shape[:2])
    zoomed = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # re-score: YOLOX on the enhanced crop (best detection near the crop centre), else blob contrast
    after = {"cls": c["cls"], "score": c["score"], "source": "unchanged"}
    det = get_detector() if use_yolox else None
    zc = np.array([zoomed.shape[1] / 2, zoomed.shape[0] / 2])
    if det is not None:
        best = None
        for d in det.detect(zoomed):
            bx, by, bw, bh = d.box
            if bx <= zc[0] <= bx + bw and by <= zc[1] <= by + bh and (best is None or d.score > best.score):
                best = d
        if best is not None and (best.score >= c["score"] or best.cls != c["cls"]):
            after = {"cls": best.cls, "score": round(best.score, 3), "source": "yolox_on_crop"}
    if after["source"] == "unchanged" and c["cls"] == "unknown":
        # local contrast of the box against a ring around it, measured on the enhanced, zoomed crop
        g = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bx0, by0 = int((x - x0) * scale), int((y - y0) * scale)
        bx1, by1 = int((x + w - x0) * scale), int((y + h - y0) * scale)
        inner = np.zeros(g.shape, bool)
        inner[max(0, by0 + (by1 - by0) // 4):by1 - (by1 - by0) // 4, max(0, bx0 + (bx1 - bx0) // 4):bx1 - (bx1 - bx0) // 4] = True
        box = np.zeros(g.shape, bool)
        box[max(0, by0):by1, max(0, bx0):bx1] = True
        ring = ~box
        if inner.any() and ring.any():
            darkness = float(np.median(g[ring]) - np.median(g[inner]))
            score = round(0.8 * float(np.clip(darkness / 110.0, 0, 1)), 3)
            after = {"cls": "unknown", "score": score, "source": "contrast_on_crop"}
    g0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std()
    g1 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).std()
    report = {"candidate_id": cid, "zoom": round(scale, 2), "crop_box": [x0, y0, x1 - x0, y1 - y0],
              "before": {"cls": c["cls"], "score": c["score"]}, "after": after,
              "contrast_gain": round(float(g1 / max(g0, 1e-3)), 2),
              "crop_visibility": visibility(crop)}
    return report, zoomed


# --- compare frames -----------------------------------------------------------------------------------------
def compare_frames(prev: np.ndarray, cur: np.ndarray, cands: list[dict], cid: int,
                   moving_px: float = 1.0) -> dict:
    """Dense Farneback flow; subtract the frame's median flow (camera ego-motion) and measure the
    residual motion inside the candidate box against the background. Animals move; rocks don't."""
    c = _find(cands, cid)
    if prev.shape != cur.shape:
        prev = cv2.resize(prev, (cur.shape[1], cur.shape[0]))
    scale = min(1.0, 320.0 / max(cur.shape[:2]))
    g0 = cv2.cvtColor(cv2.resize(prev, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cv2.resize(cur, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    ego = np.median(flow.reshape(-1, 2), axis=0)
    resid = flow - ego
    mag = np.linalg.norm(resid, axis=2) / scale          # back to original pixels
    x, y, w, h = (np.array(c["box"]) * scale).round().astype(int)
    x, y = max(0, x), max(0, y)
    inside = np.zeros(mag.shape, bool)
    inside[y:y + max(1, h), x:x + max(1, w)] = True
    m_in = float(np.percentile(mag[inside], 75)) if inside.any() else 0.0
    m_bg = float(np.median(mag[~inside])) if (~inside).any() else 0.0
    v = resid[inside].mean(axis=0) / scale if inside.any() else np.zeros(2)
    motion = max(0.0, m_in - m_bg)
    return {"candidate_id": cid, "motion_px": round(motion, 2), "background_px": round(m_bg, 2),
            "ego_motion_px": [round(float(ego[0] / scale), 2), round(float(ego[1] / scale), 2)],
            "direction_deg": round(float(math.degrees(math.atan2(v[1], v[0]))), 1),
            "moving": bool(motion >= moving_px)}


# --- map query ------------------------------------------------------------------------------------------------
def _bearing(dr: float, dc: float) -> float:
    """Compass bearing in degrees: north = -row, east = +col."""
    return round((math.degrees(math.atan2(dc, -dr)) + 360.0) % 360.0, 1)


def query_map(summary: dict, dcfg: DecisionConfig | None = None) -> dict:
    try:
        return _query_map(summary, dcfg)
    except ToolError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
        raise ToolError(f"malformed map summary ({type(e).__name__}: {str(e)[:120]})") from None


def _query_map(summary: dict, dcfg: DecisionConfig | None = None) -> dict:
    """Summarise the hiker's living map. `summary` schema (all optional except hiker.pos):
    {t, cell_m=5, step_s=4, hiker:{id,pos:[r,c]}, group:[{id,pos}], exits:[[r,c]],
     sightings:[{pos:[r,c], conf, t, hiker}], halflife, alert_radius, escape_threshold}"""
    dcfg = dcfg or DecisionConfig()
    if not isinstance(summary, dict) or "hiker" not in summary:
        raise ToolError("map summary needs at least {'hiker': {'pos': [row, col]}}")
    t = float(summary.get("t", 0))
    cell = float(summary.get("cell_m", 5.0))
    step_s = float(summary.get("step_s", 4.0))
    hl = float(summary.get("halflife", dcfg.evidence_halflife))
    alert = float(summary.get("alert_radius", dcfg.alert_radius))
    r0, c0 = (float(v) for v in summary["hiker"]["pos"])

    exits = []
    for e in summary.get("exits", []):
        d = math.hypot(e[0] - r0, e[1] - c0)
        exits.append({"distance_m": round(d * cell, 1), "bearing_deg": _bearing(e[0] - r0, e[1] - c0)})
    exits.sort(key=lambda e: e["distance_m"])

    sightings, risk = [], 0.0
    for s in summary.get("sightings", []):
        d = math.hypot(s["pos"][0] - r0, s["pos"][1] - c0)
        age = max(0.0, t - float(s.get("t", t)))
        wgt = float(s.get("conf", 0.0)) * 0.5 ** (age / hl)
        if d <= alert:
            risk += wgt
        sightings.append({"distance_m": round(d * cell, 1), "bearing_deg": _bearing(s["pos"][0] - r0, s["pos"][1] - c0),
                          "age_s": round(age * step_s, 1), "conf": round(float(s.get("conf", 0.0)), 3),
                          "weight": round(wgt, 3), "by_hiker": s.get("hiker")})
    sightings.sort(key=lambda s: -s["weight"])

    group = [math.hypot(g["pos"][0] - r0, g["pos"][1] - c0) * cell for g in summary.get("group", [])]
    thr = float(summary.get("escape_threshold", dcfg.escape_threshold))
    return {"nearest_exit": exits[0] if exits else None, "n_known_exits": len(exits),
            "recent_sightings": sightings[:5], "risk_near_hiker": round(risk, 3),
            "escape_threshold": thr, "alert_radius_m": round(alert * cell, 1),
            "nearest_sighting_m": min((s["distance_m"] for s in sightings), default=None),
            "farthest_teammate_m": round(max(group), 1) if group else None}


# --- helpers --------------------------------------------------------------------------------------------------
class ToolError(ValueError):
    """Bad tool arguments. Returned to the model as an error result, never raised through the loop."""


def _resize_max(img: np.ndarray, side: int) -> np.ndarray:
    s = side / max(img.shape[:2])
    return img if s >= 1 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _check_frame(bgr: Any) -> None:
    if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ToolError("frame must be a BGR uint8 image")
    if max(bgr.shape[:2]) > MAX_FRAME_SIDE or min(bgr.shape[:2]) < 16:
        raise ToolError(f"frame size {bgr.shape[:2]} out of range")


def annotate(bgr: np.ndarray, cands: list[dict]) -> np.ndarray:
    """Set-of-marks overlay: each candidate's box and ID drawn on the frame, so the model can refer to IDs."""
    img = bgr.copy()
    for c in cands:
        x, y, w, h = c["box"]
        col = (0, 165, 255) if c["cls"] != "person" else (255, 191, 0)   # orange / sky blue (colour-blind safe)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
        label = f"#{c['id']} {c['cls']} {c['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y - 4 if y - th - 6 > 0 else y + h + th + 4
        cv2.rectangle(img, (x, ty - th - 3), (x + tw + 4, ty + 3), (0, 0, 0), -1)
        cv2.putText(img, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def encode_image(bgr: np.ndarray, fmt: str = "jpeg", max_side: int = 512, quality: int = 85) -> bytes:
    img = _resize_max(bgr, max_side)
    ext = ".jpg" if fmt == "jpeg" else ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpeg" else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ToolError("image encoding failed")
    return buf.tobytes()


def data_url(bgr: np.ndarray, max_side: int = 512) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(encode_image(bgr, "jpeg", max_side)).decode()


# --- stateful toolbox ----------------------------------------------------------------------------------------
class Toolbox:
    """Bounded frame store + candidate cache. Frames are referenced by id; paths are only accepted
    under `allowed_roots` (no arbitrary file reads from tool arguments)."""

    def __init__(self, use_yolox: bool = True, allowed_roots: list[str | Path] | None = None, max_frames: int = 64):
        self.use_yolox = use_yolox
        self.allowed_roots = [Path(r).resolve() for r in (allowed_roots or [])]
        self.max_frames = max_frames
        self.frames: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cands: dict[str, dict] = {}
        self._closer: dict[tuple[str, int], tuple[dict, np.ndarray]] = {}
        self._lock = threading.Lock()

    # frames
    def add_frame(self, bgr: np.ndarray, frame_id: str | None = None) -> str:
        _check_frame(bgr)
        fid = frame_id or uuid.uuid4().hex[:12]
        with self._lock:
            if fid in self.frames:                       # same id, new image: stale candidates/crops must go
                self.cands.pop(fid, None)
                self._closer = {k: v for k, v in self._closer.items() if k[0] != fid}
            self.frames[fid] = bgr
            self.frames.move_to_end(fid)
            while len(self.frames) > self.max_frames:
                old, _ = self.frames.popitem(last=False)
                self.cands.pop(old, None)
                self._closer = {k: v for k, v in self._closer.items() if k[0] != old}
        return fid

    def load_frame(self, path: str | Path, frame_id: str | None = None) -> str:
        p = Path(path).resolve()
        if self.allowed_roots and not any(p.is_relative_to(r) for r in self.allowed_roots):
            raise ToolError("path outside the allowed frame roots")
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            raise ToolError("unsupported image type")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ToolError("could not read image")
        return self.add_frame(img, frame_id or p.stem)

    def frame(self, fid: str) -> np.ndarray:
        try:
            return self.frames[fid]
        except KeyError:
            raise ToolError(f"unknown frame_id {fid!r}") from None

    # tools
    def set_candidates(self, fid: str, candidates: list[dict], frame_info: dict | None = None) -> None:
        img = self.frame(fid)
        clean, seen = [], set()
        for c in candidates:
            try:
                cid, box = int(c["id"]), [float(v) for v in c["box"]]
                if len(box) != 4 or box[2] <= 0 or box[3] <= 0 or cid in seen:
                    raise ValueError
                clipped = _clip_box(box, img.shape)
            except (KeyError, TypeError, ValueError):
                raise ToolError(f"bad candidate {str(c)[:120]}: need unique int id and box [x, y, w, h]") from None
            if clipped[0] >= img.shape[1] or clipped[1] >= img.shape[0]:
                raise ToolError(f"candidate {cid} box lies outside the frame")
            seen.add(cid)
            clean.append({**c, "id": cid, "cls": str(c.get("cls", "unknown")), "score": float(c.get("score", 0.0)),
                          "box": clipped, "source": c.get("source", "given")})
        self.cands[fid] = {"frame": frame_info or visibility(img), "detector": "given", "candidates": clean}

    def detect(self, fid: str) -> dict:
        if fid not in self.cands:
            self.cands[fid] = detect_candidates(self.frame(fid), use_yolox=self.use_yolox)
        return self.cands[fid]

    def candidates(self, fid: str) -> list[dict]:
        return self.detect(fid)["candidates"]

    def look_closer(self, fid: str, cid: int, zoom: float = 2.5) -> tuple[dict, np.ndarray]:
        key = (fid, int(cid))
        if key not in self._closer:
            self._closer[key] = look_closer(self.frame(fid), self.candidates(fid), int(cid), zoom, self.use_yolox)
        return self._closer[key]

    def compare(self, prev_fid: str, fid: str, cid: int) -> dict:
        return compare_frames(self.frame(prev_fid), self.frame(fid), self.candidates(fid), int(cid))

    def query_map(self, summary: dict) -> dict:
        return query_map(summary)

    def annotated(self, fid: str) -> np.ndarray:
        return annotate(self.frame(fid), self.candidates(fid))

    def reset_tool_cache(self) -> None:
        """Forget memoised look_closer results (so each policy pays its own tool latency)."""
        self._closer.clear()

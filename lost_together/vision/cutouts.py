"""Animal and hiker cut-outs: OpenCV grabCut segmentation of permissively licensed photos into RGBA PNGs.

Sources (URL, author, licence) are listed in data/cutouts/SOURCES.md and data/cutouts/manifest.json.
Each cut-out is initialised from a hand-placed rectangle (fractions of the photo), segmented with
cv2.grabCut, then cleaned with morphology (open/close, largest connected component, hole filling)
and a slightly feathered alpha edge.

    uv run python -m lost_together.vision.cutouts --fetch  # download source photos + licence metadata
    uv run python -m lost_together.vision.cutouts          # build cut-outs + YOLOX sanity check
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CUTDIR = ROOT / "data" / "cutouts"

# real-world heights (m) of the object as it appears (shoulder/back height for quadrupeds incl. head)
HEIGHT_M = {"bear": 1.0, "person": 1.75, "deer": 1.1, "dog": 0.6}


@dataclass(frozen=True)
class Spec:
    id: str
    cls: str                 # bear | person | deer | dog
    rect: tuple              # x0, y0, x1, y1 as fractions of the source photo
    coco: str                # COCO class YOLOX should report (decoys: the class we *fear* it reports)
    bg_rule: str = ""        # optional colour prior marking definite background: "snow" | "grass"


SPECS = (
    Spec("bear_grizzly_snow", "bear", (0.215, 0.255, 0.74, 0.77), "bear", "snow"),
    Spec("bear_grizzly_rocks", "bear", (0.175, 0.40, 0.575, 0.875), "bear"),
    Spec("person_hiker_desert", "person", (0.78, 0.355, 0.865, 0.59), "person"),
    Spec("person_hiker_path", "person", (0.275, 0.365, 0.365, 0.69), "person"),
    Spec("deer_doe_standing", "deer", (0.30, 0.18, 0.735, 0.955), "bear", "snow"),
    Spec("deer_doe_grazing", "deer", (0.285, 0.165, 0.685, 0.765), "bear", "snow"),
    Spec("dog_shepherd", "dog", (0.055, 0.085, 0.875, 0.945), "dog", "grass"),
)
# Wikimedia Commons file titles (licences verified on the file pages, 2026-10-26; see SOURCES.md)
COMMONS = {
    "bear_grizzly_snow": "File:Grizzly bear walking on top of snow near Obsidian Creek. "
                         "(d8eed9f2-b700-4f44-9c54-b9d5a0d99409).jpg",
    "bear_grizzly_rocks": "File:Grizzly bear (8703042570).jpg",
    "person_hiker_desert": "File:Visitor hiking Lake View Trail near Katherine Landing "
                           "(15c577da-cce7-4057-9c85-9b077a6f2358).jpg",
    "person_hiker_path": "File:Walking River Mountains Loop Trail (a04683af-d98e-46a6-8573-8f5e353b55c4).jpg",
    "deer_doe_standing": "File:White-tailed deer (53485642462).jpg",
    "deer_doe_grazing": "File:White-tailed deer (53485642157).jpg",
    "dog_shepherd": "File:German-shepherd-4055031920.jpg",
}
# Retired cut-outs: the id still resolves (to a public-domain replacement, with a warning) so callers outside
# this package keep working, but the photo is no longer shipped or used in calibration.
#   bear_black_cdfw: Commons tags it CC BY 2.0 (Flickr, CDFW), but its own description calls it a
#   "Corel rights-free image", i.e. third-party stock the uploader may not have been able to license -> not clearly
#   CC BY. Removed on review (2026-10-26).
RETIRED = {"bear_black_cdfw": "bear_grizzly_rocks"}
BEARS = tuple(s.id for s in SPECS if s.cls == "bear")
PERSONS = tuple(s.id for s in SPECS if s.cls == "person")
DECOYS = tuple(s.id for s in SPECS if s.cls in ("deer", "dog"))


def segment(img: np.ndarray, rect_frac, work_px: int = 700, iters: int = 6, bg_rule: str = ""):
    """grabCut inside a rectangle, then morphology clean-up. Returns (crop BGR, alpha uint8)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rect_frac
    rx0, ry0, rx1, ry1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    # work on a padded crop around the rectangle, resized so the object is ~work_px on its long side
    pad = int(0.08 * max(rx1 - rx0, ry1 - ry0))
    cx0, cy0, cx1, cy1 = max(0, rx0 - pad), max(0, ry0 - pad), min(w, rx1 + pad), min(h, ry1 + pad)
    crop = img[cy0:cy1, cx0:cx1]
    s = work_px / max(rx1 - rx0, ry1 - ry0)
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    rect = (int((rx0 - cx0) * s), int((ry0 - cy0) * s), int((rx1 - rx0) * s), int((ry1 - ry0) * s))
    mask = np.zeros(crop.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    if bg_rule:
        # second pass: pixels matching a background colour prior become definite background
        c = crop.astype(np.int16)
        b, g, r = c[..., 0], c[..., 1], c[..., 2]
        if bg_rule == "snow":      # bright snow and blue snow shadow
            bgm = (b - r > 45) | (c.min(-1) > 195)
        else:                      # green grass / yellow flowers
            bgm = ((g > r + 8) & (g > b + 8)) | ((r > 170) & (g > 150) & (b < 90))
        bgm = cv2.morphologyEx(bgm.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        mask[bgm] = cv2.GC_BGD
        cv2.grabCut(crop, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = clean_mask(fg)
    ys, xs = np.nonzero(fg)
    y0b, y1b, x0b, x1b = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, fg = crop[y0b:y1b, x0b:x1b], fg[y0b:y1b, x0b:x1b]
    alpha = cv2.GaussianBlur(fg, (3, 3), 0)          # soften the edge by ~1 px
    return crop, alpha


def clean_mask(fg: np.ndarray) -> np.ndarray:
    k = max(3, (min(fg.shape) // 120) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ker)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ker, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = np.where(lab == big, 255, 0).astype(np.uint8)
    # fill interior holes: flood the background from the border, everything not reached is object
    ff = np.pad(fg, 1)
    m = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 128)
    fg = np.where(ff[1:-1, 1:-1] == 128, 0, 255).astype(np.uint8)
    return fg


def fetch_sources(width: int = 2400):
    """Download each photo (Commons-resized to `width`) and its author/licence metadata (idempotent)."""
    import re
    import urllib.parse
    import urllib.request
    ua = {"User-Agent": "lost-together/0.1 (OpenCV AI Competition 2026)"}
    (CUTDIR / "src").mkdir(parents=True, exist_ok=True)
    meta = {}
    for sp in SPECS:
        q = urllib.parse.urlencode({"action": "query", "format": "json", "titles": COMMONS[sp.id],
                                    "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": width})
        with urllib.request.urlopen(urllib.request.Request(f"https://commons.wikimedia.org/w/api.php?{q}",
                                                           headers=ua), timeout=60) as r:
            page = next(iter(json.load(r)["query"]["pages"].values()))
        ii = page["imageinfo"][0]
        m = ii["extmetadata"]
        g = lambda f: re.sub(r"\s+", " ", re.sub("<[^>]+>", "", m.get(f, {}).get("value", ""))).strip()  # noqa
        out = CUTDIR / "src" / f"{sp.id}.jpg"
        if not out.exists():
            with urllib.request.urlopen(urllib.request.Request(ii["thumburl"], headers=ua), timeout=120) as r:
                out.write_bytes(r.read())
        meta[sp.id] = {"class": sp.cls, "title": COMMONS[sp.id], "page": ii["descriptionurl"],
                       "original_url": ii["url"], "downloaded_url": ii["thumburl"], "author": g("Artist"),
                       "credit": g("Credit")[:200], "license": g("LicenseShortName"),
                       "license_url": g("LicenseUrl"), "usage_terms": g("UsageTerms")[:200],
                       "original_size": [ii["width"], ii["height"]]}
    (CUTDIR / "sources_raw.json").write_text(json.dumps(meta, indent=1))
    return meta


def resolve(cutout_id: str) -> str:
    if cutout_id in RETIRED:
        import warnings
        warnings.warn(f"cut-out {cutout_id!r} is retired (licence not clear); using {RETIRED[cutout_id]!r}",
                      stacklevel=3)
        return RETIRED[cutout_id]
    return cutout_id


def load(cutout_id: str) -> np.ndarray:
    """RGBA cut-out as BGRA uint8 (tight crop: bottom row = ground contact)."""
    cutout_id = resolve(cutout_id)
    im = cv2.imread(str(CUTDIR / f"{cutout_id}.png"), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"{CUTDIR / cutout_id}.png missing; run python -m lost_together.vision.cutouts")
    return im


def cls_of(cutout_id: str) -> str:
    cutout_id = RETIRED.get(cutout_id, cutout_id)
    return next(s.cls for s in SPECS if s.id == cutout_id)


def on_neutral(bgra: np.ndarray, height_px: int = 300, size: int = 640, grey: int = 150) -> np.ndarray:
    """Paste a cut-out on a flat grey canvas (sanity check for the detector)."""
    s = height_px / bgra.shape[0]
    obj = cv2.resize(bgra, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if obj.shape[1] > size - 40:
        s2 = (size - 40) / obj.shape[1]
        obj = cv2.resize(obj, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), grey, np.uint8)
    h, w = obj.shape[:2]
    y, x = (size - h) // 2, (size - w) // 2
    a = obj[:, :, 3:4].astype(np.float32) / 255
    canvas[y:y + h, x:x + w] = (a * obj[:, :, :3] + (1 - a) * canvas[y:y + h, x:x + w]).astype(np.uint8)
    return canvas


def build(sanity: bool = True):
    from .detector import YoloxDetector
    raw = json.loads((CUTDIR / "sources_raw.json").read_text())
    manifest = {}
    det = YoloxDetector(conf=0.05) if sanity else None
    sheet = []
    for sp in SPECS:
        img = cv2.imread(str(CUTDIR / "src" / f"{sp.id}.jpg"))
        crop, alpha = segment(img, sp.rect, bg_rule=sp.bg_rule)
        bgra = np.dstack([crop, alpha])
        cv2.imwrite(str(CUTDIR / f"{sp.id}.png"), bgra)
        r = raw[sp.id]
        entry = {"class": sp.cls, "height_m": HEIGHT_M[sp.cls], "file": f"{sp.id}.png",
                 "size_px": [int(bgra.shape[1]), int(bgra.shape[0])],
                 "alpha_fill": round(float((alpha > 127).mean()), 3), "init_rect_frac": sp.rect,
                 "source_page": r["page"], "source_file_url": r["original_url"],
                 "downloaded_url": r["downloaded_url"], "author": r["author"], "credit": r["credit"],
                 "license": r["license"], "license_url": r["license_url"]}
        if det is not None:
            neutral = on_neutral(bgra)
            dets = det.detect(neutral)
            top = {}
            for d in dets:
                top[d.cls] = max(top.get(d.cls, 0.0), round(d.score, 3))
            entry["yolox_on_neutral"] = dict(sorted(top.items(), key=lambda kv: -kv[1])[:4])
            entry["yolox_top_class"] = dets[0].cls if dets else None
            vis = neutral.copy()
            for d in dets[:3]:
                x, y, w, h = map(int, d.box)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(vis, f"{d.cls} {d.score:.2f}", (x, max(14, y - 4)), 0, 0.6, (0, 0, 255), 2)
            cv2.putText(vis, sp.id, (8, 24), 0, 0.7, (255, 255, 255), 2)
            sheet.append(cv2.resize(vis, (320, 320)))
        manifest[sp.id] = entry
        print(sp.id, entry["size_px"], entry.get("yolox_on_neutral"))
    (CUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if sheet:
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        cv2.imwrite(str(CUTDIR / "sanity_sheet.jpg"),
                    np.vstack([np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet), 4)]))
    return manifest


if __name__ == "__main__":
    import sys
    cv2.setNumThreads(4)
    if "--fetch" in sys.argv:
        fetch_sources()
    build()

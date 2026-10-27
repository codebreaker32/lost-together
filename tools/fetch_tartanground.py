"""Fetch the TartanGround ForestEnv frames the vision calibration needs (idempotent, cached).

TartanGround (Patel et al., arXiv:2505.10696), CC BY 4.0, https://tartanair.org/tartanground/.
Zips on Hugging Face are STORED, so each member is one HTTP Range request (no full-zip download).

  (a) background frames: ~40 frames spread over ForestEnv omni/P0004, omni/P0015, omni/P0009, diff/P1009
      (image_lcam_front, image_rcam_front, depth_lcam_front, seg_lcam_front)
  (b) odometry run: consecutive P0004 frames 0..120 (image_lcam_front, image_rcam_front, depth_lcam_front)

Layout: data/tartanground/ForestEnv_<traj>/<modality>/<frame>_<cam>_<...>.png, plus
data/tartanground/backgrounds.json listing the background frames.

usage: uv run python tools/fetch_tartanground.py [--backgrounds-only|--odometry-only] [--workers 6]
"""
import argparse
import io
import json
import struct
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tartanground"
BASE = "https://huggingface.co/datasets/theairlabcmu/TartanGround/resolve/main"
UA = {"User-Agent": "lost-together-fetch/0.1"}

BG_MODS = ("image_lcam_front", "image_rcam_front", "depth_lcam_front", "seg_lcam_front")
ODO_MODS = ("image_lcam_front", "image_rcam_front", "depth_lcam_front")
# (data dir, trajectory, number of frames, background frames to sample)
TRAJS = (("Data_omni", "P0004", 502, 12), ("Data_omni", "P0015", 1739, 10),
         ("Data_omni", "P0009", 2840, 10), ("Data_diff", "P1009", 644, 10))
ODO_TRAJ, ODO_FRAMES = ("Data_omni", "P0004"), range(0, 121)


def member_name(mod: str, idx: int) -> str:
    kind, cam, side = mod.split("_", 2)   # image|depth|seg, lcam|rcam, front
    suffix = "" if kind == "image" else f"_{kind}"
    return f"{mod}/{idx:06d}_{cam}_{side}{suffix}.png"


class RangeReader(io.RawIOBase):
    """Just enough of a seekable file for zipfile to read the central directory."""

    def __init__(self, url):
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url, self.size = r.geturl(), int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = get_range(self.url, self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data


def get_range(url, start, end, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:   # noqa: BLE001 - retry transient network errors
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def extract(url, infos: list[zipfile.ZipInfo], outdir: Path):
    """Read STORED members with one range request each (local header + data)."""
    outdir.mkdir(parents=True, exist_ok=True)
    for zi in infos:
        if zi.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{zi.filename} is compressed; expected STORED")
        out = outdir / Path(zi.filename).name   # flatten, never trust archive paths
        if out.exists() and out.stat().st_size == zi.file_size:
            continue
        blob = get_range(url, zi.header_offset, zi.header_offset + 30 + 512 + zi.compress_size)
        if blob[:4] != b"PK\x03\x04":
            raise ValueError(f"bad local header for {zi.filename}")
        nlen, xlen = struct.unpack("<HH", blob[26:30])
        data = blob[30 + nlen + xlen: 30 + nlen + xlen + zi.compress_size]
        if len(data) != zi.file_size or zipfile.crc32(data) != zi.CRC:
            raise ValueError(f"size/CRC mismatch for {zi.filename}")
        tmp = out.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(out)


def fetch(data_dir, traj, mod, frames, log):
    outdir = DATA / f"ForestEnv_{traj}" / mod
    want = {member_name(mod, i) for i in frames}
    if all((outdir / Path(w).name).exists() for w in want):
        return 0
    url = f"{BASE}/ForestEnv/{data_dir}/{traj}/{mod}.zip"
    rr = RangeReader(url)
    z = zipfile.ZipFile(rr)
    infos = [zi for zi in z.infolist() if zi.filename in want]
    missing = want - {zi.filename for zi in infos}
    if missing:
        log(f"  warning: {len(missing)} members not in {url}, e.g. {sorted(missing)[:2]}")
    extract(rr.url, infos, outdir)
    log(f"  {traj}/{mod}: {len(infos)} frames")
    return len(infos)


def background_frames():
    out = []
    for data_dir, traj, n, k in TRAJS:
        # spread evenly, skipping the first/last 5% (start-up and stop poses)
        lo, hi = int(0.05 * n), int(0.95 * n)
        idx = [lo + round(j * (hi - lo) / (k - 1)) for j in range(k)]
        out += [{"data_dir": data_dir, "traj": traj, "frame": i} for i in idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backgrounds-only", action="store_true")
    ap.add_argument("--odometry-only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)   # noqa: E731
    jobs = []
    if not a.odometry_only:
        bgs = background_frames()
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "backgrounds.json").write_text(json.dumps(bgs, indent=1))
        for data_dir, traj, _, _ in TRAJS:
            frames = [b["frame"] for b in bgs if b["traj"] == traj]
            jobs += [(data_dir, traj, m, frames) for m in BG_MODS]
    if not a.backgrounds_only:
        jobs += [(*ODO_TRAJ, m, list(ODO_FRAMES)) for m in ODO_MODS]
    t0 = time.time()
    with ThreadPoolExecutor(min(a.workers, 6)) as ex:
        futs = [ex.submit(fetch, *j, log) for j in jobs]
        errs = [f.exception() for f in futs if f.exception()]
    for e in errs:
        log(f"error: {e!r}")
    # pose files for the background trajectories (already cached under data/tartanground/poses)
    log(f"done in {time.time() - t0:.0f}s; {len(errs)} errors")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())

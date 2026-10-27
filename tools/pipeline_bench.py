"""Per-frame perception pipeline benchmark: the same code and frames on any box (stock pip OpenCV, COOL AMI, x86).

  uv run python tools/pipeline_bench.py --label stock-graviton-c8g.2xlarge --out results/bench_stock_graviton.json
  # on the COOL AMI (its own cv2; do not pip-install opencv-python there):
  PYTHONPATH=. python3 tools/pipeline_bench.py --label cool-graviton --out results/bench_cool_graviton.json

Stages per hiker frame: composite a bear into a TartanGround stereo pair with fog (render), candidate
detection (detect: YOLOX-s on the new DNN engine, also timed alone as yolox, plus dark blobs and haze), StereoSGBM depth (stereo), and the agent's look_closer / compare_frames tools.
Each stage is timed separately at 1 thread and at all threads. cv2.getBuildInformation() is saved as proof
of which OpenCV build ran.
"""
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from lost_together.agent import tools as T
from lost_together.vision import composite as C
from lost_together.vision.detector import YoloxDetector
from lost_together.vision.stereo import StereoDepth


def timed(fn, reps):
    fn()                                                   # warm-up
    ms = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t) * 1000)
    return {"median_ms": round(statistics.median(ms), 2), "p90_ms": round(float(np.percentile(ms, 90)), 2),
            "reps": reps}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--frames", type=int, default=4, help="background frames to cycle through")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--usd-per-hour", type=float, help="instance on-demand price, for $ per 1,000 frames")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)

    bgs = json.loads((C.TG / "backgrounds.json").read_text())[: a.frames]
    frames = [C.load_frame(b["traj"], b["frame"]) for b in bgs]
    det = YoloxDetector()
    sd = StereoDepth()
    state = {"i": 0}

    def render():
        f = frames[state["i"] % len(frames)]
        state["i"] += 1
        return C.render(f, [("bear_grizzly_rocks", 12.0, 0.0)], beta=0.05, stereo=True)

    r = render()
    left, right = r["left"], r["right"]
    prev = C.render(frames[0], [("bear_grizzly_rocks", 13.0, 1.0)], beta=0.05, stereo=False)["left"]
    cands = T.detect_candidates(left)["candidates"]
    cid = cands[0]["id"] if cands else None

    stages = {
        "render": render,
        "yolox": lambda: det.detect(left),
        "detect": lambda: T.detect_candidates(left),       # YOLOX + dark blobs + haze estimate
        "stereo": lambda: sd.compute(left, right),
    }
    if cid is not None:
        stages["look_closer"] = lambda: T.look_closer(left, cands, cid)
        stages["compare_frames"] = lambda: T.compare_frames(prev, left, cands, cid)

    out = {"label": a.label, "cv2_version": cv2.__version__, "cv2_file": cv2.__file__,
           "machine": platform.machine(), "processor": platform.processor(), "cpus": os.cpu_count(),
           "python": platform.python_version(), "frame_shape": list(left.shape), "results": {}}
    for threads in (1, os.cpu_count()):
        cv2.setNumThreads(threads)
        res = {k: timed(fn, a.reps) for k, fn in stages.items()}
        frame_ms = sum(res[k]["median_ms"] for k in ("render", "detect", "stereo"))
        res["per_frame_ms"] = round(frame_ms, 2)
        res["frames_per_s"] = round(1000 / frame_ms, 2)
        if a.usd_per_hour:
            # one process per thread-budget: at 1 thread, run cpus processes in parallel
            fps_box = res["frames_per_s"] * (os.cpu_count() if threads == 1 else 1)
            res["usd_per_1000_frames"] = round(a.usd_per_hour / 3600 / fps_box * 1000, 5)
        out["results"][f"threads_{threads}"] = res
        print(a.label, f"threads={threads}", json.dumps(res))
    out["build_information"] = cv2.getBuildInformation()
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

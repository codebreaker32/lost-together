"""Synthetic escalation points for offline tests and a pipeline sanity set (NOT a Tier C result).

A frame is forest-like noise with trunks, a dark shape (predator) or a grey rock (decoy) or nothing,
and fog from the scattering model I = J*exp(-beta*d) + A*(1 - exp(-beta*d)). The predator moves
between the previous and the current frame; a rock doesn't. Truth labels go only into `truth`,
which the policies never see.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

W, H = 480, 360


def forest_background(rng: np.random.Generator) -> np.ndarray:
    base = np.empty((H, W, 3), np.float32)
    base[...] = (60, 110, 80)                                   # BGR: muted green
    noise = rng.normal(0, 1, (H // 8, W // 8, 3)).astype(np.float32)
    noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC) * 22
    img = base + noise + rng.normal(0, 8, (H, W, 3)).astype(np.float32)
    for _ in range(int(rng.integers(3, 6))):                    # trunks: tall, thin, brownish
        x = int(rng.integers(0, W - 20))
        tw = int(rng.integers(8, 18))
        img[:, x:x + tw] = img[:, x:x + tw] * 0.55 + np.array((40, 60, 75), np.float32) * 0.45
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_shape(img: np.ndarray, kind: str, center: tuple[int, int], size: int, rng: np.random.Generator) -> None:
    cx, cy = center
    if kind == "predator":                                       # dark, bulky body + head
        cv2.ellipse(img, (cx, cy), (size, int(size * 0.62)), 0, 0, 360, (25, 28, 32), -1, cv2.LINE_AA)
        cv2.circle(img, (cx + int(size * 0.9), cy - int(size * 0.3)), int(size * 0.38), (22, 25, 30), -1, cv2.LINE_AA)
    elif kind == "rock":                                         # grey, flat, lower contrast
        cv2.ellipse(img, (cx, cy), (size, int(size * 0.5)), 0, 0, 360, (48, 54, 58), -1, cv2.LINE_AA)


def fog(img: np.ndarray, beta: float, depth_m: float = 25.0, airlight: int = 215) -> np.ndarray:
    t = np.exp(-beta * depth_m)
    return np.clip(img.astype(np.float32) * t + airlight * (1 - t), 0, 255).astype(np.uint8)


def make_point(i: int, rng: np.random.Generator, kind: str, out_dir: Path) -> dict:
    beta = float(rng.choice([0.0, 0.02, 0.04]))
    size = int(rng.integers(26, 46))
    cx, cy = int(rng.integers(80, W - 100)), int(rng.integers(140, H - 60))
    bg = forest_background(rng)
    prev, cur = bg.copy(), bg.copy()
    dx = int(rng.integers(6, 12)) * (1 if rng.random() < 0.5 else -1) if kind == "predator" else 0
    draw_shape(prev, kind, (cx - dx, cy), size, rng)
    draw_shape(cur, kind, (cx, cy), size, rng)
    prev, cur = fog(prev, beta), fog(cur, beta)
    pd, cd = out_dir / f"p{i:04d}_prev.png", out_dir / f"p{i:04d}.png"
    cv2.imwrite(str(pd), prev)
    cv2.imwrite(str(cd), cur)

    hiker = [40, 40]
    sightings = []
    if rng.random() < (0.7 if kind == "predator" else 0.3):    # a teammate's earlier report nearby
        sightings.append({"pos": [hiker[0] + int(rng.integers(-8, 9)), hiker[1] + int(rng.integers(-8, 9))],
                          "conf": round(float(rng.uniform(0.35, 0.8)), 2), "t": 100 - int(rng.integers(0, 30)),
                          "hiker": 1})
    summary = {"t": 100, "cell_m": 5.0, "step_s": 4.0, "hiker": {"id": 0, "pos": hiker},
               "group": [{"id": j, "pos": [hiker[0] + int(rng.integers(-15, 16)), hiker[1] + int(rng.integers(-15, 16))]}
                         for j in (1, 2, 3)],
               "exits": [[0, int(rng.integers(0, 80))], [79, int(rng.integers(0, 80))]],
               "sightings": sightings}
    return {"point_id": f"synth-{i:04d}", "frame_path": cd.name, "prev_frame_path": pd.name,
            "candidates": None, "map_summary": summary,
            "truth": {"predator_real": kind == "predator", "should_escape": kind == "predator", "kind": kind,
                      "fog_beta": beta}}


def make_dataset(out_dir: str | Path, n: int = 12, seed: int = 0) -> Path:
    """Writes frames and `points.jsonl` (frame paths relative to the jsonl). Returns the jsonl path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    kinds = ["predator", "rock", "none"]
    path = out / "points.jsonl"
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps(make_point(i, rng, kinds[i % 3], out)) + "\n")
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write a synthetic escalation-point set (pipeline sanity only).")
    ap.add_argument("--out", default="results/agent_synth")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_dataset(a.out, a.n, a.seed))

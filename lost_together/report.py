"""Hypothesis report for a calibrated Tier A sweep, faceted by fog (β). Same method as the v0 write-up:
paired-by-seed differences per (speed, radius) cell, 95% bootstrap CI, counts of cells above/below 0.

  uv run python -m lost_together.report results/tierA_v1.csv --ref results/tierA_ref.csv \
      --group results/tierA_v1_group.csv --control results/tierA_v1_off.csv --out results/tierA_v1_report.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analysis import bootstrap_ci, load


def index(rows, beta=None):
    """{(speed, radius): {seed: {coop: frac_escaped}}}, optionally for one β."""
    out = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if beta is not None and float(r.get("beta") or 0.0) != beta:
            continue
        out[(float(r["speed_ratio"]), float(r["sense_radius"]))][r["seed"]][r["cooperation"]] = float(r["frac_escaped"])
    return out


def contrast(a, b, coop_a, coop_b):
    """Per-cell paired Δ = a[coop_a] − b[coop_b] over seeds present in both."""
    cells = {}
    for cell in sorted(set(a) & set(b)):
        d = np.array([a[cell][s][coop_a] - b[cell][s][coop_b] for s in a[cell]
                      if coop_a in a[cell][s] and s in b[cell] and coop_b in b[cell][s]])
        if len(d):
            lo, hi = bootstrap_ci(d)
            cells[cell] = {"delta": float(d.mean()), "lo": lo, "hi": hi, "n": len(d)}
    return cells


def summary(cells):
    deltas = [c["delta"] for c in cells.values()]
    by_speed = defaultdict(list)
    by_radius = defaultdict(list)
    for (s, r), c in cells.items():
        by_speed[s].append(c["delta"])
        by_radius[r].append(c["delta"])
    return {
        "n_cells": len(cells),
        "above0": sum(c["lo"] > 0 for c in cells.values()),
        "below0": sum(c["hi"] < 0 for c in cells.values()),
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "min_delta": float(np.min(deltas)) if deltas else None,
        "max_delta": float(np.max(deltas)) if deltas else None,
        "by_speed": {f"{k:g}": float(np.mean(v)) for k, v in sorted(by_speed.items())},
        "by_radius": {f"{k:g}": float(np.mean(v)) for k, v in sorted(by_radius.items())},
    }


def grid_mean(idx, coop):
    v = [x[coop] for seeds in idx.values() for x in seeds.values() if coop in x]
    return float(np.mean(v)) if v else None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--ref", type=Path, help="perfect-perception sweep (β = 0) on the same grid and seeds")
    p.add_argument("--group", type=Path, help="group-vs-share sweep")
    p.add_argument("--control", type=Path, help="predator-off sweep")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    rows = load(args.csv)
    betas = sorted({float(r.get("beta") or 0.0) for r in rows})
    ref = index(load(args.ref)) if args.ref else None
    report = {"source": str(args.csv), "betas": {}}
    for beta in betas:
        idx = index(rows, beta)
        b = {"means": {c: grid_mean(idx, c) for c in ("solo", "sightings", "map", "share")}}
        for coop in ("share", "sightings", "map"):
            b[f"{coop}-solo"] = summary(contrast(idx, idx, coop, "solo"))
        if ref is not None:
            b["h4_vs_perfect"] = {c: summary(contrast(idx, ref, c, c)) for c in ("solo", "sightings", "map", "share")}
            v_solo, v_share, p_share = b["means"]["solo"], b["means"]["share"], grid_mean(ref, "share")
            b["perfect_share_mean"] = p_share
            b["gap_recovered"] = (v_share - v_solo) / (p_share - v_solo) if p_share != v_solo else None
        report["betas"][f"{beta:g}"] = b
    if args.group:
        g = index(load(args.group))
        report["group-share"] = summary(contrast(g, g, "group", "share"))
        report["group_means"] = {c: grid_mean(g, c) for c in ("group", "share")}
    if args.control:
        off = load(args.control)
        ctrl = {}
        for beta in sorted({float(r.get("beta") or 0.0) for r in off}):
            o = index(off, beta)
            ctrl[f"{beta:g}"] = {"means": {c: grid_mean(o, c) for c in ("solo", "sightings", "map", "share", "group")},
                                 "sightings-solo": summary(contrast(o, o, "sightings", "solo")),
                                 "share-solo": summary(contrast(o, o, "share", "solo"))}
        report["control_off"] = ctrl
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

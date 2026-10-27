"""Headline figure: where does sharing help? Paired Δ escape rate vs solo, three kinds of sharing, one colour scale.

  uv run python -m lost_together.figures results/tierA_v0.csv --control results/tierA_v0_off.csv
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import bootstrap_ci, load, paired_deltas  # noqa: E402

PANELS = [("share", "Share everything"), ("sightings", "Share predator sightings only"), ("map", "Share terrain + exits only")]


def grid_for(rows, coop):
    d = paired_deltas(rows, coop)
    speeds = sorted({k[-2] for k in d})
    radii = sorted({k[-1] for k in d})
    g = np.full((len(radii), len(speeds)), np.nan)
    sig = np.zeros_like(g, bool)
    for k, x in d.items():
        i, j = radii.index(k[-1]), speeds.index(k[-2])
        lo, hi = bootstrap_ci(x)
        g[i, j], sig[i, j] = x.mean(), lo > 0 or hi < 0
    return speeds, radii, g, sig, len(next(iter(d.values())))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--control", type=Path, help="predator-off sweep CSV")
    p.add_argument("--beta", type=float, help="keep only rows with this fog coefficient (calibrated sweeps)")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    rows = load(args.csv)
    if args.beta is not None:
        rows = [r for r in rows if float(r.get("beta") or 0.0) == args.beta]
    panels = [(c, t) for c, t in PANELS if any(r["cooperation"] == c for r in rows)]
    grids = [grid_for(rows, c) for c, _ in panels]
    lim = max(np.nanmax(np.abs(g[2])) for g in grids)

    ncol = len(panels) + (1 if args.control else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (coop, title), (speeds, radii, g, sig, n) in zip(axes, panels, grids):
        im = ax.imshow(g, origin="lower", cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(len(radii)):
            for j in range(len(speeds)):
                if not sig[i, j]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", ec="0.75", lw=0))
                ax.text(j, i, f"{g[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                        color="black" if sig[i, j] else "0.5")
        ax.set_xticks(range(len(speeds)), [f"{s:g}" for s in speeds])
        ax.set_yticks(range(len(radii)), [f"{r * 5:g}" for r in radii])
        ax.set_xlabel("predator speed ÷ hiker speed")
        ax.set_ylabel("predator sensing radius (m)")
        ax.set_title(f"{title}\nΔ escape rate vs solo ({n} paired seeds/cell)", fontsize=9)
    fig.colorbar(im, ax=axes[:len(panels)], shrink=0.85, label="Δ escape rate vs solo")
    fog = f"fog β = {args.beta:g} /m · " if args.beta is not None else ""
    fig.suptitle(f"{fog}Hatched cells: 95% bootstrap CI of the paired difference includes 0", fontsize=8, color="0.35")

    if args.control:
        off = load(args.control)
        if args.beta is not None:
            off = [r for r in off if float(r.get("beta") or 0.0) == args.beta]
        ax = axes[-1]
        order = [c for c in ("solo", "sightings", "map", "share", "group")
                 if any(r["cooperation"] == c for r in off) and any(r["cooperation"] == c for r in rows)]
        on_means = [np.mean([float(r["frac_escaped"]) for r in rows if r["cooperation"] == c]) for c in order]
        off_means = [np.mean([float(r["frac_escaped"]) for r in off if r["cooperation"] == c]) for c in order]
        x = np.arange(len(order))
        ax.bar(x - 0.2, off_means, 0.4, label="predator off", color="#9ecae1")
        ax.bar(x + 0.2, on_means, 0.4, label="predator on (grid mean)", color="#3182bd")
        ax.set_xticks(x, order)
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of hikers escaping")
        ax.set_title("Control: what does sharing buy\nwithout a predator?", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
    out = args.out or args.csv.with_name(f"{args.csv.stem}_headline.png")
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

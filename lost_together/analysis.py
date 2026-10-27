"""Phase diagram from a sweep CSV: paired difference in escape rate, cooperation minus solo, per cell.

  uv run python -m lost_together.analysis results/pilot1.csv
Writes results/<name>_phase.png and prints a table. Cells whose 95% bootstrap CI includes 0 are hatched.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AXES = ("speed_ratio", "sense_radius")
FACETS = ("perception", "fog", "beta", "miss_prob", "false_cue_rate", "predator_on", "policy", "comm_range")


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def paired_deltas(rows, coop: str, base: str = "solo", metric: str = "frac_escaped"):
    """{(facet..., speed, radius): array of per-seed (coop - base) differences}."""
    by = defaultdict(dict)
    for r in rows:
        key = tuple(r.get(k, "") for k in FACETS) + tuple(float(r[a]) for a in AXES)
        by[(key, r["seed"])][r["cooperation"]] = float(r[metric])
    out = defaultdict(list)
    for (key, _seed), v in by.items():
        if coop in v and base in v:
            out[key].append(v[coop] - v[base])
    return {k: np.array(v) for k, v in out.items()}


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarise(rows, coop: str):
    table = []
    for key, d in sorted(paired_deltas(rows, coop).items()):
        lo, hi = bootstrap_ci(d)
        table.append({"facet": key[:len(FACETS)], "speed_ratio": key[-2], "sense_radius": key[-1],
                      "delta": float(d.mean()), "lo": lo, "hi": hi, "n": len(d)})
    return table


def plot(table, coop: str, out: Path, title: str) -> None:
    speeds = sorted({t["speed_ratio"] for t in table})
    radii = sorted({t["sense_radius"] for t in table})
    grid = np.full((len(radii), len(speeds)), np.nan)
    sig = np.zeros_like(grid, bool)
    for t in table:
        i, j = radii.index(t["sense_radius"]), speeds.index(t["speed_ratio"])
        grid[i, j] = t["delta"]
        sig[i, j] = t["lo"] > 0 or t["hi"] < 0
    lim = np.nanmax(np.abs(grid)) or 1.0
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(grid, origin="lower", cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(radii)):
        for j in range(len(speeds)):
            ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color="black" if sig[i, j] else "0.45", fontweight="bold" if sig[i, j] else "normal")
            if not sig[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", ec="0.7", lw=0))
    if np.isfinite(grid).sum() >= 4 and np.nanmin(grid) < 0 < np.nanmax(grid):
        ax.contour(np.arange(len(speeds)), np.arange(len(radii)), grid, levels=[0], colors="k", linewidths=1.5)
    ax.set_xticks(range(len(speeds)), [f"{s:g}" for s in speeds])
    ax.set_yticks(range(len(radii)), [f"{r:g}" for r in radii])
    ax.set_xlabel("predator speed / hiker speed")
    ax.set_ylabel("predator sensing radius (cells, 5 m each)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label=f"Δ escape rate ({coop} − solo)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--coop", nargs="+", default=["share", "group"])
    args = p.parse_args(argv)
    rows = load(args.csv)
    for coop in args.coop:
        table = summarise(rows, coop)
        if not table:
            continue
        facets = sorted({t["facet"] for t in table})
        for facet in facets:
            sub = [t for t in table if t["facet"] == facet]
            tag = "_".join(f"{k}{v}" for k, v in zip(FACETS, facet) if k != "predator_on")
            out = args.csv.with_name(f"{args.csv.stem}_{coop}_{tag}.png")
            plot(sub, coop, out, f"{coop} vs solo · {', '.join(f'{k}={v}' for k, v in zip(FACETS, facet))}\n"
                                 f"hatched: 95% CI includes 0 (n={sub[0]['n']} paired seeds)")
            print(f"\n{coop} - solo  [{', '.join(f'{k}={v}' for k, v in zip(FACETS, facet))}]  -> {out.name}")
            print("speed  radius   delta    95% CI")
            for t in sub:
                print(f"{t['speed_ratio']:5g}  {t['sense_radius']:6g}  {t['delta']:+.3f}  [{t['lo']:+.3f}, {t['hi']:+.3f}]")
    print("\nmean escape rate by cooperation:")
    for coop in ("solo", "share", "group"):
        vals = [float(r["frac_escaped"]) for r in rows if r["cooperation"] == coop]
        if vals:
            print(f"  {coop:5s} {np.mean(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()

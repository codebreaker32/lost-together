"""Parallel parameter sweep. Every (cell, cooperation) runs the same seeds, so results pair by seed.

  uv run python -m lost_together.sweep --name pilot --speed-ratio 0.5 1 1.5 --sense-radius 10 20 30 \
      --coop solo share group --seeds 40
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from .config import Cooperation, EpisodeConfig, Perception, PredatorPolicy
from .sim import run_episode

RESULTS = Path(__file__).resolve().parent.parent / "results"


def build_configs(args) -> list[EpisodeConfig]:
    base = EpisodeConfig()
    out = []
    for sr, rad, miss, fog, beta, fcr, coop, perc, pred_on, policy, comm in itertools.product(
            args.speed_ratio, args.sense_radius, args.miss_prob, args.fog, args.beta, args.false_cue_rate,
            args.coop, args.perception, args.predator_on, args.policy, args.comm_range):
        out.append(replace(
            base,
            cooperation=Cooperation(coop),
            predator_on=pred_on,
            comm=replace(base.comm, range_cells=None if comm <= 0 else comm),
            predator=replace(base.predator, speed_ratio=sr, sense_radius=rad, miss_prob=miss,
                             policy=PredatorPolicy(policy)),
            perception=replace(base.perception, mode=Perception(perc), fog=fog, beta=beta, false_cue_rate=fcr,
                               calibration=args.calibration),
        ))
    return out


def _run(job):
    cfg, seed = job
    try:
        return run_episode(cfg, seed), None
    except Exception:
        return None, f"seed={seed} cooperation={cfg.cooperation.value} predator={cfg.predator} " \
                     f"perception={cfg.perception}\n{traceback.format_exc()}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--speed-ratio", type=float, nargs="+", default=[1.0])
    p.add_argument("--sense-radius", type=float, nargs="+", default=[20.0])
    p.add_argument("--miss-prob", type=float, nargs="+", default=[0.3])
    p.add_argument("--fog", type=float, nargs="+", default=[0.0])
    p.add_argument("--beta", type=float, nargs="+", default=[0.0], help="fog extinction per metre (calibrated mode)")
    p.add_argument("--calibration", default=str(RESULTS / "calibration.json"))
    p.add_argument("--false-cue-rate", type=float, nargs="+", default=[0.01])
    p.add_argument("--coop", nargs="+", default=[c.value for c in Cooperation])
    p.add_argument("--perception", nargs="+", default=[Perception.SURROGATE.value])
    p.add_argument("--predator-on", type=lambda s: s.lower() in ("1", "true", "yes"), nargs="+", default=[True])
    p.add_argument("--policy", nargs="+", default=[PredatorPolicy.PURSUE.value], choices=[x.value for x in PredatorPolicy])
    p.add_argument("--comm-range", type=float, nargs="+", default=[0.0], help="cells; 0 = global instant sharing")
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args(argv)

    configs = build_configs(args)
    jobs = [(c, s) for c in configs for s in range(args.seed_offset, args.seed_offset + args.seeds)]
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.name}.csv"
    (RESULTS / f"{args.name}.json").write_text(json.dumps(
        {"args": vars(args), "base_config": asdict(EpisodeConfig()), "n_jobs": len(jobs)}, indent=2, default=str))

    t0, done, errors = time.time(), 0, []
    with out.open("w", newline="") as f, ProcessPoolExecutor(args.workers) as pool:
        writer = None
        for fut in as_completed([pool.submit(_run, j) for j in jobs]):
            row, err = fut.result()
            if err:
                errors.append(err)
                continue
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            done += 1
            if done % max(1, len(jobs) // 20) == 0:
                el = time.time() - t0
                print(f"{done}/{len(jobs)} episodes, {el:.0f}s elapsed, ~{el / done * (len(jobs) - done):.0f}s left", flush=True)
    print(f"wrote {out} ({done} episodes, {time.time() - t0:.0f}s)")
    if errors:
        err_path = RESULTS / f"{args.name}.errors.txt"
        err_path.write_text("\n\n".join(errors))
        sys.exit(f"{len(errors)} episodes failed; see {err_path}")


if __name__ == "__main__":
    main()

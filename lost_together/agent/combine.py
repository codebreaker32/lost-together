"""Combine Tier C bench runs (e.g. two halves run in parallel) into one report, raw and re-weighted.

The bench points are stratified (more real sightings than occur naturally), so rates are also reported
re-weighted to the natural real fraction: accuracy_w = f·acc(real) + (1−f)·acc(false), and false alarms /
late escapes per 100 natural decision moments.

  uv run python -m lost_together.agent.combine results/agent_bench/tierC_live_a results/agent_bench/tierC_live_b \
      --natural-real 0.183 --out results/agent_bench/tierC_live_report.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load(dirs):
    rows = defaultdict(dict)                       # point_id -> policy -> record
    for d in dirs:
        for line in (Path(d) / "decisions.jsonl").open():
            r = json.loads(line)
            rows[r["point_id"]][r["policy"]] = r
    return rows


def correct(r):
    return (r["decision"] == "escape") == bool(r["truth"]["should_escape"])


def beta_of(pid: str) -> str:
    return pid.rsplit("_b", 1)[-1]


def score(recs, f_real, boot_seed=0):
    recs = list(recs)
    ok = np.array([correct(r) for r in recs])
    should = np.array([bool(r["truth"]["should_escape"]) for r in recs])
    esc = np.array([r["decision"] == "escape" for r in recs])
    lat = np.array([r["latency_s"] for r in recs])
    mlat = np.array([r["model_latency_s"] for r in recs])
    fa_rate = float((esc & ~should).sum() / max(1, (~should).sum()))
    late_rate = float((~esc & should).sum() / max(1, should.sum()))
    acc_real = float(ok[should].mean()) if should.any() else float("nan")
    acc_false = float(ok[~should].mean()) if (~should).any() else float("nan")
    return {
        "n": len(recs), "n_should_escape": int(should.sum()),
        "accuracy": round(float(ok.mean()), 4),
        "false_alarms": int((esc & ~should).sum()), "false_alarm_rate": round(fa_rate, 4),
        "late_escapes": int((~esc & should).sum()), "late_escape_rate": round(late_rate, 4),
        "accuracy_reweighted": round(f_real * acc_real + (1 - f_real) * acc_false, 4),
        "false_alarms_per_100_natural": round(100 * (1 - f_real) * fa_rate, 2),
        "late_escapes_per_100_natural": round(100 * f_real * late_rate, 2),
        "fallback_rate": round(float(np.mean([r["source"] == "fallback" for r in recs])), 4),
        "decisions": dict(zip(*np.unique([r["decision"] for r in recs], return_counts=True))),
        "latency_p50_s": round(float(np.percentile(lat, 50)), 3), "latency_p95_s": round(float(np.percentile(lat, 95)), 3),
        "model_latency_p50_s": round(float(np.percentile(mlat, 50)), 3),
        "model_latency_p95_s": round(float(np.percentile(mlat, 95)), 3),
        "model_seconds_total": round(float(mlat.sum()), 1),
    }


def mcnemar(rows, a, b):
    pairs = [(correct(v[a]), correct(v[b])) for v in rows.values() if a in v and b in v]
    b10 = sum(x and not y for x, y in pairs)
    b01 = sum(y and not x for x, y in pairs)
    nd = b10 + b01
    return {f"{a}_right_{b}_wrong": b10, f"{a}_wrong_{b}_right": b01, "discordant": nd,
            "p_exact": round(float(stats.binomtest(min(b10, b01), nd, 0.5).pvalue), 6) if nd else 1.0}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--natural-real", type=float, default=0.183)
    p.add_argument("--usd-per-hour", type=float, default=2.235)
    p.add_argument("--gpu-wall-s", type=float, help="GPU wall-clock seconds spent on these runs (for amortised $)")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    rows = load(a.dirs)
    policies = sorted({k for v in rows.values() for k in v})
    rep = {"runs": a.dirs, "n_points": len(rows), "natural_real_fraction": a.natural_real, "policies": {},
           "by_beta": {}}
    for pol in policies:
        recs = [v[pol] for v in rows.values() if pol in v]
        s = score(recs, a.natural_real)
        if pol != "rule":
            s["usd_per_decision_model_time"] = round(s["model_seconds_total"] / s["n"] * a.usd_per_hour / 3600, 5)
            if a.gpu_wall_s:
                s["usd_per_decision_gpu_wall"] = round(a.gpu_wall_s * a.usd_per_hour / 3600 / s["n"], 5)
        rep["policies"][pol] = s
        for beta in sorted({beta_of(k) for k in rows}):
            sub = [v[pol] for k, v in rows.items() if pol in v and beta_of(k) == beta]
            rep["by_beta"].setdefault(beta, {})[pol] = {k: score(sub, a.natural_real)[k] for k in
                                                         ("n", "accuracy", "accuracy_reweighted", "false_alarms",
                                                          "late_escapes")}
    for other in policies:
        if other != "rule":
            rep[f"mcnemar_rule_vs_{other}"] = mcnemar(rows, "rule", other)
    if "llm" in policies and "llm_scalar" in policies:
        rep["mcnemar_llm_vs_llm_scalar"] = mcnemar(rows, "llm", "llm_scalar")
    text = json.dumps(rep, indent=1, default=int)
    if a.out:
        a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

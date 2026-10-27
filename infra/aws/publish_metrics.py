"""Publish a sweep's summary metrics to CloudWatch (namespace LostTogether) and create the dashboard.

  uv run python infra/aws/publish_metrics.py results/pilot3.csv            # dry run (default): prints payloads
  uv run python infra/aws/publish_metrics.py results/pilot3.csv --apply    # put-metric-data + put-dashboard

Stdlib only (boto3 is not a dependency): it shells out to the AWS CLI v2, so it also runs on the COOL box.
Per (sweep, perception, predator on/off, cooperation mode) it publishes:
  EscapeRate            mean share of hikers who escaped, % (the headline metric)
  AllEscapedRate        % of episodes in which every hiker escaped
  DeltaVsSolo           paired-by-seed escape-rate difference vs solo, percentage points (non-solo modes)
  DeltaVsSoloCILow/High 95% bootstrap CI of DeltaVsSolo
  Episodes              number of episodes (rows)
  FalseAlarms           total false alarm decisions
  FalseAlarmsPerEpisode false alarms per episode
  CaughtPerEpisode      hikers intercepted per episode
  CellsWhereModeHelps/Hurts, Cells  cells whose per-cell Δ 95% CI lies above / below 0, and the cell count
and per sweep: TotalEpisodes.
Timestamps are "now" unless --timestamp is given; CloudWatch accepts points up to two weeks old.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

NAMESPACE = "LostTogether"
HERE = Path(__file__).resolve().parent
CELL_AXES = ("speed_ratio", "sense_radius", "miss_prob", "fog", "false_cue_rate", "n_hikers")
MAX_DATUMS_PER_CALL = 500  # API limit is 1,000 datums / 1 MB per request


# ---------------------------------------------------------------- summary (pure, testable)

def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choices(xs, k=k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[min(n - 1, int(0.975 * n))]


def group_key(r: dict) -> tuple[str, str]:
    pred = "on" if str(r.get("predator_on", "True")).lower() in ("true", "1") else "off"
    return r.get("perception", "unknown"), pred


def summarise(rows: list[dict], base: str = "solo") -> dict:
    """{(perception, predator): {mode: {metric: value}}} plus per-sweep figures under key None."""
    by_mode = defaultdict(lambda: defaultdict(list))
    paired = defaultdict(dict)  # (group, cell, seed) -> {mode: frac_escaped}
    for r in rows:
        g = group_key(r)
        by_mode[g][r["cooperation"]].append(r)
        cell = tuple(r.get(a, "") for a in CELL_AXES)
        paired[(g, cell, r["seed"])][r["cooperation"]] = _f(r["frac_escaped"])

    deltas = defaultdict(list)       # (group, mode) -> per-seed deltas
    cell_deltas = defaultdict(list)  # (group, mode, cell) -> per-seed deltas
    for (g, cell, _seed), v in paired.items():
        if base not in v:
            continue
        for mode, x in v.items():
            if mode != base:
                deltas[(g, mode)].append(x - v[base])
                cell_deltas[(g, mode, cell)].append(x - v[base])

    out: dict = {}
    for g, modes in by_mode.items():
        out[g] = {}
        for mode, rs in sorted(modes.items()):
            m = {
                "EscapeRate": 100 * _mean(_f(r["frac_escaped"]) for r in rs),
                "AllEscapedRate": 100 * _mean(1.0 if r.get("all_escaped") == "True" else 0.0 for r in rs),
                "Episodes": float(len(rs)),
                "FalseAlarms": sum(_f(r.get("n_false_alarms", 0)) for r in rs),
                "CaughtPerEpisode": _mean(_f(r.get("n_caught", "nan")) for r in rs),
            }
            m["FalseAlarmsPerEpisode"] = m["FalseAlarms"] / len(rs)
            d = deltas.get((g, mode))
            if d:
                lo, hi = bootstrap_ci(d)
                m.update(DeltaVsSolo=100 * _mean(d), DeltaVsSoloCILow=100 * lo,
                         DeltaVsSoloCIHigh=100 * hi)
                helps = sum(1 for (gg, mm, _c), cd in cell_deltas.items()
                            if gg == g and mm == mode and len(cd) >= 5 and bootstrap_ci(cd, n=500)[0] > 0)
                hurts = sum(1 for (gg, mm, _c), cd in cell_deltas.items()
                            if gg == g and mm == mode and len(cd) >= 5 and bootstrap_ci(cd, n=500)[1] < 0)
                m["CellsWhereModeHelps"] = float(helps)
                m["CellsWhereModeHurts"] = float(hurts)
                m["Cells"] = float(sum(1 for (gg, mm, _c) in cell_deltas if gg == g and mm == mode))
            out[g][mode] = m
    out[None] = {"TotalEpisodes": float(len(rows))}
    return out


UNITS = {"EscapeRate": "Percent", "AllEscapedRate": "Percent", "DeltaVsSolo": "None",
         "DeltaVsSoloCILow": "None", "DeltaVsSoloCIHigh": "None", "Episodes": "Count",
         "FalseAlarms": "Count", "FalseAlarmsPerEpisode": "None", "CaughtPerEpisode": "None",
         "CellsWhereModeHelps": "Count", "CellsWhereModeHurts": "Count", "Cells": "Count",
         "TotalEpisodes": "Count"}


def to_metric_data(summary: dict, sweep: str, ts: str) -> list[dict]:
    data = []
    for g, modes in summary.items():
        if g is None:
            for name, v in modes.items():
                data.append({"MetricName": name, "Dimensions": [{"Name": "Sweep", "Value": sweep}],
                             "Timestamp": ts, "Value": round(v, 4), "Unit": UNITS[name]})
            continue
        perception, pred = g
        for mode, metrics in modes.items():
            dims = [{"Name": "Sweep", "Value": sweep}, {"Name": "Perception", "Value": perception},
                    {"Name": "Predator", "Value": pred}, {"Name": "Cooperation", "Value": mode}]
            for name, v in metrics.items():
                if name not in UNITS or v != v:  # skip helpers and NaN (CloudWatch rejects NaN)
                    continue
                data.append({"MetricName": name, "Dimensions": dims, "Timestamp": ts,
                             "Value": round(v, 4), "Unit": UNITS[name]})
    return data


def chunks(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


# ---------------------------------------------------------------- dashboard

def dashboard_body(region: str, distribution_id: str | None) -> dict:
    body = json.loads((HERE / "dashboard.json").read_text())
    widgets = []
    for w in body["widgets"]:
        props = w.get("properties", {})
        if not distribution_id and "Replay site" in props.get("markdown", ""):
            continue  # section header for the CloudFront widgets below
        if props.get("region") == "us-east-1" and "${DISTRIBUTION_ID}" in json.dumps(props):
            if not distribution_id:
                continue  # CloudFront widgets need the deployed distribution
            w = json.loads(json.dumps(w).replace("${DISTRIBUTION_ID}", distribution_id))
        elif "region" in props:
            props["region"] = region
        widgets.append(w)
    body["widgets"] = widgets
    return body


# ---------------------------------------------------------------- AWS CLI

def aws(args: list[str], profile: str, region: str) -> str:
    cmd = ["aws", "--profile", profile, "--region", region, *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"aws {' '.join(args[:2])} failed: {res.stderr.strip()}")
    return res.stdout.strip()


def stack_output(stack: str, key: str, profile: str, region: str) -> str | None:
    cmd = ["aws", "--profile", profile, "--region", region, "cloudformation", "describe-stacks",
           "--stack-name", stack, "--query", f"Stacks[0].Outputs[?OutputKey=='{key}'].OutputValue", "--output", "text"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    v = res.stdout.strip()
    return v if res.returncode == 0 and v and v != "None" else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="sweep CSV, e.g. results/pilot3.csv")
    ap.add_argument("--sweep", help="Sweep dimension value (default: CSV file stem)")
    ap.add_argument("--apply", action="store_true", help="really call AWS (default is a dry run)")
    ap.add_argument("--dry-run", action="store_true", help="print payloads only (the default)")
    ap.add_argument("--no-dashboard", action="store_true", help="skip the create_dashboard step")
    ap.add_argument("--dashboard-name", default="LostTogether")
    ap.add_argument("--distribution-id", help="CloudFront id for the site widgets (default: read from the site stack)")
    ap.add_argument("--stack-name", default=os.environ.get("STACK_NAME", "lost-together-site"))
    ap.add_argument("--timestamp", help="ISO-8601 UTC timestamp for all points (default: now)")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "default"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-south-1"))
    ap.add_argument("--expect-account-suffix", default=os.environ.get("EXPECTED_ACCOUNT_SUFFIX", "8661"),
                    help="refuse to --apply in any other account ('' to disable)")
    a = ap.parse_args(argv)
    if a.apply and a.dry_run:
        ap.error("--apply and --dry-run are mutually exclusive")
    dry = not a.apply

    with a.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{a.csv}: no rows")
    sweep = a.sweep or a.csv.stem
    ts = a.timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary = summarise(rows)
    data = to_metric_data(summary, sweep, ts)

    # Human-readable table first.
    print(f"# sweep={sweep} rows={len(rows)} namespace={NAMESPACE} region={a.region} "
          f"mode={'DRY RUN' if dry else 'APPLY'}", file=sys.stderr)
    for g, modes in summary.items():
        if g is None:
            continue
        print(f"# perception={g[0]} predator={g[1]}", file=sys.stderr)
        print(f"#   {'mode':<10} {'episodes':>8} {'escape%':>8} {'Δ vs solo (pp) [95% CI]':>26} "
              f"{'false alarms/ep':>16} {'cells +/-':>10}", file=sys.stderr)
        for mode, m in modes.items():
            d = (f"{m['DeltaVsSolo']:+6.1f} [{m['DeltaVsSoloCILow']:+.1f}, {m['DeltaVsSoloCIHigh']:+.1f}]"
                 if "DeltaVsSolo" in m else "-")
            cells = f"{m['CellsWhereModeHelps']:.0f}/{m['CellsWhereModeHurts']:.0f}" if "Cells" in m else "-"
            print(f"#   {mode:<10} {m['Episodes']:>8.0f} {m['EscapeRate']:>8.1f} {d:>26} "
                  f"{m['FalseAlarmsPerEpisode']:>16.2f} {cells:>10}", file=sys.stderr)

    if not dry and a.expect_account_suffix:
        acct = aws(["sts", "get-caller-identity", "--query", "Account", "--output", "text"], a.profile, a.region)
        if not acct.endswith(a.expect_account_suffix):
            sys.exit(f"profile {a.profile} is account ...{acct[-4:]}, expected ...{a.expect_account_suffix}")

    for i, batch in enumerate(chunks(data, MAX_DATUMS_PER_CALL)):
        payload = json.dumps(batch)
        if dry:
            payload = "[\n " + ",\n ".join(json.dumps(d, ensure_ascii=False) for d in batch) + "\n]"
            print(f"\n## put-metric-data call {i + 1}: aws --profile {a.profile} --region {a.region} "
                  f"cloudwatch put-metric-data --namespace {NAMESPACE} --metric-data file://payload.json")
            print(payload)
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tf.write(payload)
        try:
            aws(["cloudwatch", "put-metric-data", "--namespace", NAMESPACE, "--metric-data", f"file://{tf.name}"],
                a.profile, a.region)
        finally:
            os.unlink(tf.name)
    print(f"# {'would publish' if dry else 'published'} {len(data)} datapoints", file=sys.stderr)

    if not a.no_dashboard:  # create_dashboard step
        dist = a.distribution_id or (None if dry else stack_output(a.stack_name, "DistributionId", a.profile, a.region))
        body = dashboard_body(a.region, dist)
        if dry:
            print(f"\n## create_dashboard: aws cloudwatch put-dashboard --dashboard-name {a.dashboard_name} "
                  f"--dashboard-body file://body.json  ({len(body['widgets'])} widgets"
                  f"{'' if dist else '; CloudFront widgets added once the site stack exists'})")
            print(json.dumps(body, indent=1))
        else:
            res = aws(["cloudwatch", "put-dashboard", "--dashboard-name", a.dashboard_name,
                       "--dashboard-body", json.dumps(body), "--output", "json"], a.profile, a.region)
            problems = json.loads(res or "{}").get("DashboardValidationMessages", [])
            for p in problems:
                print(f"# dashboard warning: {p}", file=sys.stderr)
            print(f"# dashboard: https://{a.region}.console.aws.amazon.com/cloudwatch/home"
                  f"?region={a.region}#dashboards/dashboard/{a.dashboard_name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

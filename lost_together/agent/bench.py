"""Tier C harness: the rule-based policy and the LLM policy decide on the identical escalation points.

Input JSONL, one escalation point per line (paths relative to the file):
    {"point_id", "frame_path", "prev_frame_path"?, "candidates"?: [{id, cls, score, box}] | null,
     "map_summary": {...}, "truth": {"predator_real": bool, "should_escape": bool}}
`truth` is read only by the scorer here, never by a policy.

Scores per policy: escape accuracy, false alarms (escape when it should not), late escapes (no escape when it
should), fallback rate, tool use, p50/p95 latency and $/decision. Paired comparison: McNemar's exact test on
per-point correctness. The rule's evidence is swept over thresholds to give its false-alarm vs late-escape
curve; the LLM is one point against it. Ablation `--ablation scalar`: the LLM gets no image and no tools.

    uv run python -m lost_together.agent.synth --out results/agent_synth -n 30            # sanity set
    uv run python -m lost_together.agent.bench results/agent_synth/points.jsonl --mode stub --name synth_stub
    uv run python -m lost_together.agent.bench POINTS.jsonl --mode live   # GPU session (records the cache)
    uv run python -m lost_together.agent.bench POINTS.jsonl --mode replay # offline re-run from the cache
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from ..config import DecisionConfig
from . import tools as T
from .llm import LLMClient, LLMConfig
from .loop import AgentConfig, Decision, LocalExecutor, Point, agent_decide, load_points, rule_decide, write_trace
from .ranger import Autonomy, Ranger, make_approver

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BenchConfig:
    name: str = "bench"
    out_dir: Path | str = ROOT / "results" / "agent_bench"
    autonomy: str = "advise"
    approver: str = "auto"
    ablation: str = "none"                # none | scalar
    repeat: int = 1                       # k-repeat run-to-run variance for the LLM
    use_yolox: bool = True
    tools: str = "local"                  # local | mcp
    save_images: bool = False


# --- scoring -------------------------------------------------------------------------------------------------
def score(decisions: list[Decision], truths: dict[str, dict]) -> dict:
    n = len(decisions)
    esc = np.array([d.decision == "escape" for d in decisions])
    should = np.array([bool(truths[d.point_id]["should_escape"]) for d in decisions])
    real = np.array([bool(truths[d.point_id].get("predator_real", truths[d.point_id]["should_escape"])) for d in decisions])
    alert = np.array([d.decision in ("escape", "regroup") for d in decisions])
    lat = np.array([d.latency_s for d in decisions])
    mlat = np.array([d.model_latency_s for d in decisions])
    fa, late = int((esc & ~should).sum()), int((~esc & should).sum())
    tools = Counter(s["name"] for d in decisions for s in d.trace if s["kind"] == "tool")
    pct = lambda a, q: round(float(np.percentile(a, q)), 4) if len(a) else None   # noqa: E731
    return {
        "n": n,
        "accuracy": round(float((esc == should).mean()), 4) if n else None,
        "false_alarms": fa, "false_alarm_rate": round(fa / max(1, int((~should).sum())), 4),
        "late_escapes": late, "late_escape_rate": round(late / max(1, int(should.sum())), 4),
        "threat_recognition_accuracy": round(float((alert == real).mean()), 4) if n else None,
        "decisions": dict(Counter(d.decision for d in decisions)),
        "actions_after_ranger": dict(Counter(d.action or d.decision for d in decisions)),
        "ranger_overrides": sum(1 for d in decisions if d.approval and not d.approval["approved"]),
        "fallback_rate": round(sum(d.source == "fallback" for d in decisions) / max(1, n), 4),
        "fallback_reasons": dict(Counter(d.fallback_reason for d in decisions if d.fallback_reason)),
        "tool_calls_mean": round(float(np.mean([d.tool_calls for d in decisions])), 3) if n else 0,
        "tool_usage": dict(tools),
        "llm_calls_mean": round(float(np.mean([d.llm_calls for d in decisions])), 3) if n else 0,
        "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95),
        "model_latency_p50_s": pct(mlat, 50), "model_latency_p95_s": pct(mlat, 95),
        "usd_per_decision": round(float(np.mean([d.cost_usd for d in decisions])), 6) if n else 0,
        "tokens_in_mean": round(float(np.mean([d.tokens_in for d in decisions])), 1) if n else 0,
        "tokens_out_mean": round(float(np.mean([d.tokens_out for d in decisions])), 1) if n else 0,
    }


def mcnemar(a: list[Decision], b: list[Decision], truths: dict[str, dict]) -> dict:
    """Paired test on per-point correctness. b10 = A right & B wrong, b01 = A wrong & B right."""
    ok = lambda d: (d.decision == "escape") == bool(truths[d.point_id]["should_escape"])   # noqa: E731
    bmap = {d.point_id: d for d in b}
    b10 = sum(1 for d in a if ok(d) and not ok(bmap[d.point_id]))
    b01 = sum(1 for d in a if not ok(d) and ok(bmap[d.point_id]))
    nd = b10 + b01
    p_exact = float(stats.binomtest(min(b10, b01), nd, 0.5).pvalue) if nd else 1.0
    chi2 = (abs(b10 - b01) - 1) ** 2 / nd if nd else 0.0
    return {"rule_right_llm_wrong": b10, "rule_wrong_llm_right": b01, "discordant": nd,
            "p_exact": round(p_exact, 6), "chi2_cc": round(chi2, 4),
            "p_chi2": round(float(stats.chi2.sf(chi2, 1)), 6) if nd else 1.0}


def rule_curve(rule: list[Decision], truths: dict[str, dict], thresholds=None) -> list[dict]:
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.1, 2.01, 0.1), 2)
    ev = np.array([d.evidence or 0.0 for d in rule])
    should = np.array([bool(truths[d.point_id]["should_escape"]) for d in rule])
    out = []
    for t in thresholds:
        esc = ev >= t
        out.append({"threshold": float(t),
                    "false_alarm_rate": round(float((esc & ~should).sum() / max(1, (~should).sum())), 4),
                    "late_escape_rate": round(float((~esc & should).sum() / max(1, should.sum())), 4)})
    return out


def plot_curve(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = report["rule_curve"]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot([p["false_alarm_rate"] for p in c], [p["late_escape_rate"] for p in c], "-o", ms=3, color="#0072B2",
            label="rule (threshold sweep)")
    marks = {"rule": ("s", "#0072B2"), "llm": ("*", "#D55E00"), "llm_scalar": ("^", "#009E73")}
    for k, (m, col) in marks.items():
        if k in report["policies"]:
            p = report["policies"][k]
            ax.plot(p["false_alarm_rate"], p["late_escape_rate"], m, ms=12 if m == "*" else 8, color=col, label=k)
    ax.set_xlabel("false-alarm escape rate")
    ax.set_ylabel("late-escape rate")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    tag = " - STUB, NOT A RESULT" if report["mode"] == "stub" else ""
    ax.set_title(f"{report['name']} ({report['mode']}, n={report['n']}){tag}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- run -----------------------------------------------------------------------------------------------------
def run(points: list[Point], llm_cfg: LLMConfig, bcfg: BenchConfig, acfg: AgentConfig | None = None,
        dcfg: DecisionConfig | None = None, stub=None, points_path: Path | None = None) -> dict:
    out = Path(bcfg.out_dir) / bcfg.name
    (out / "traces").mkdir(parents=True, exist_ok=True)
    acfg = acfg or AgentConfig()
    if bcfg.save_images:
        acfg.artifacts_dir = out / "img"
    tb = T.Toolbox(use_yolox=bcfg.use_yolox)
    client = LLMClient(llm_cfg, stub=stub)
    ranger = Ranger(Autonomy(bcfg.autonomy), make_approver(bcfg.approver), log_path=out / "ranger_log.jsonl")
    variants = {"llm": acfg}
    if bcfg.ablation == "scalar":
        variants["llm_scalar"] = AgentConfig(**{**acfg.__dict__, "send_image": False, "allow_tools": False})
    session = None
    if bcfg.tools == "mcp":
        import sys

        from .mcp_server import McpExecutor, McpSession
        if points_path is None:
            raise ValueError("MCP tools need file-backed points")
        cmd = [sys.executable, "-m", "lost_together.agent.mcp_server", "--root", str(Path(points_path).resolve().parent)]
        session = McpSession(command=cmd + ([] if bcfg.use_yolox else ["--no-yolox"]))
    results: dict[str, list[Decision]] = {"rule": [], **{k: [] for k in variants}}
    repeats: dict[str, list[str]] = {}
    truths = {p.point_id: p.truth for p in points}
    perception_s: list[float] = []
    t_start = time.time()
    try:
        with (out / "decisions.jsonl").open("w") as f:
            for i, p in enumerate(points):
                if p.truth is None:
                    raise ValueError(f"point {p.point_id} has no truth label")
                if session is not None:
                    if points_path is None:
                        raise ValueError("MCP tools need file-backed points")
                    rec = _raw_record(points_path, p.point_id)
                    base = points_path.parent
                    ex = McpExecutor(session, base / rec["frame_path"], p.map_summary,
                                     base / rec["prev_frame_path"] if rec.get("prev_frame_path") else None)
                else:
                    ex = LocalExecutor(tb, p)
                t_obs = time.perf_counter()
                ex.observe()                                   # shared perception, timed separately
                perception_s.append(time.perf_counter() - t_obs)
                blind = dataclasses.replace(p, truth=None)    # policies never get the label
                ds = [rule_decide(ex, p.point_id, acfg, dcfg)]
                for name, vcfg in variants.items():
                    tb.reset_tool_cache()
                    client.cfg.sample = 0
                    d = agent_decide(blind, client, ex, vcfg, dcfg)
                    d.policy = name
                    ds.append(d)
                    if name == "llm" and bcfg.repeat > 1:
                        reps = [d.decision]
                        for k in range(1, bcfg.repeat):
                            client.cfg.sample = k
                            reps.append(agent_decide(blind, client, ex, vcfg, dcfg).decision)
                        repeats[p.point_id] = reps
                        client.cfg.sample = 0
                for d in ds:
                    ranger.review(d)
                    results[d.policy].append(d)
                    write_trace(d, out / "traces")
                    f.write(json.dumps({"point_id": p.point_id, "policy": d.policy, "decision": d.decision,
                                        "action": d.action, "candidate_id": d.candidate_id, "why": d.why,
                                        "source": d.source, "fallback_reason": d.fallback_reason,
                                        "evidence": d.evidence, "latency_s": round(d.latency_s, 4),
                                        "model_latency_s": round(d.model_latency_s, 4), "story": d.story(),
                                        "truth": p.truth}) + "\n")
                print(f"[{i + 1}/{len(points)}] {p.point_id}: " +
                      "  ".join(f"{d.policy}={d.decision}{'*' if d.source == 'fallback' else ''}" for d in ds) +
                      f"  (truth should_escape={p.truth['should_escape']})", flush=True)
    finally:
        if session is not None:
            session.close()

    report = {
        "name": bcfg.name, "mode": llm_cfg.mode, "model": llm_cfg.model if llm_cfg.mode != "stub" else "stub",
        "n": len(points), "detector": T.detector_status() if bcfg.use_yolox else "blob-only",
        "tools_via": bcfg.tools, "autonomy": bcfg.autonomy, "approver": bcfg.approver,
        "image_in_tool": llm_cfg.image_in_tool, "max_tool_turns": acfg.max_tool_turns,
        "wall_s": round(time.time() - t_start, 2),
        "perception_p50_s": round(float(np.median(perception_s)), 4) if perception_s else None,
        "latency_note": "latency_s is the decision only; shared OpenCV detection is timed as perception_p50_s. "
                        "usd_per_decision = model latency x GPU $/h (+ per-token prices if set).",
        "policies": {k: score(v, truths) for k, v in results.items()},
        "mcnemar_rule_vs_llm": mcnemar(results["rule"], results["llm"], truths),
        "rule_curve": rule_curve(results["rule"], truths),
        "usd_per_hour_assumed": llm_cfg.usd_per_hour,
    }
    if "llm_scalar" in results:
        report["mcnemar_llm_vs_scalar"] = mcnemar(results["llm_scalar"], results["llm"], truths)
    if repeats:
        agree = [len(set(r)) == 1 for r in repeats.values()]
        report["repeat"] = {"k": bcfg.repeat, "all_agree_rate": round(float(np.mean(agree)), 4)}
    if llm_cfg.mode == "stub":
        report["warning"] = "STUB MODEL: tests the pipeline only; these numbers are not a result."
    if llm_cfg.mode == "replay":
        report["note"] = "Replayed from cached responses; latency and cost are those recorded in the live session."
    (out / "report.json").write_text(json.dumps(report, indent=1))
    plot_curve(report, out / "rule_vs_llm.png")
    return report


def _raw_record(points_path: Path, point_id: str) -> dict:
    for line in points_path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if str(rec["point_id"]) == point_id:
                return rec
    raise KeyError(point_id)


def print_report(r: dict) -> None:
    cols = ["accuracy", "false_alarms", "late_escapes", "fallback_rate", "tool_calls_mean",
            "latency_p50_s", "latency_p95_s", "usd_per_decision"]
    print(f"\n{r['name']}: mode={r['mode']} model={r['model']} n={r['n']} detector={r['detector']}")
    if "warning" in r:
        print("  " + r["warning"])
    print("  " + "policy".ljust(12) + "".join(c[:14].rjust(15) for c in cols))
    for k, m in r["policies"].items():
        print("  " + k.ljust(12) + "".join(str(m[c]).rjust(15) for c in cols))
    mc = r["mcnemar_rule_vs_llm"]
    print(f"  McNemar rule vs llm: rule-only-right={mc['rule_right_llm_wrong']} llm-only-right={mc['rule_wrong_llm_right']}"
          f" p_exact={mc['p_exact']}")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Tier C: rule-based vs LLM policy on identical escalation points.")
    ap.add_argument("points", nargs="?", help="escalation points jsonl")
    ap.add_argument("--make-synthetic", type=int, metavar="N", help="first write N synthetic points (sanity set)")
    ap.add_argument("--synthetic-dir", default=str(ROOT / "results" / "agent_synth"))
    ap.add_argument("--mode", default="stub", choices=["stub", "replay", "live"])
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--key-env", help="env var holding an API key (e.g. GROQ_KEY); never printed")
    ap.add_argument("--image-in-user", action="store_true", help="send tool images in a follow-up user message (Groq)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--usd-per-hour", type=float, default=2.235)
    ap.add_argument("--name", default="bench")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "agent_bench"))
    ap.add_argument("--autonomy", default="advise", choices=["advise", "act"])
    ap.add_argument("--approver", default="auto", choices=["auto", "reject", "cli"])
    ap.add_argument("--ablation", default="none", choices=["none", "scalar"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-tool-turns", type=int, default=3)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tools", default="local", choices=["local", "mcp"])
    ap.add_argument("--no-yolox", action="store_true")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--rule-relook-low", type=float, default=0.1,
                    help="lower edge of the rule's look-closer band; 0.1 = the candidate threshold the agent sees")
    a = ap.parse_args(argv)
    if a.make_synthetic:
        from .synth import make_dataset
        a.points = str(make_dataset(a.synthetic_dir, a.make_synthetic))
    if not a.points:
        ap.error("points jsonl required (or --make-synthetic N)")
    pts = load_points(a.points)[: a.limit]
    llm_cfg = LLMConfig.from_env(mode=a.mode, base_url=a.base_url, model=a.model, api_key_env=a.key_env,
                                 timeout_s=a.timeout, usd_per_hour=a.usd_per_hour)
    if a.image_in_user:
        llm_cfg.image_in_tool = False
    bcfg = BenchConfig(name=a.name, out_dir=a.out_dir, autonomy=a.autonomy, approver=a.approver,
                       ablation=a.ablation, repeat=a.repeat, use_yolox=not a.no_yolox, tools=a.tools,
                       save_images=a.save_images)
    report = run(pts, llm_cfg, bcfg, AgentConfig(max_tool_turns=a.max_tool_turns), points_path=Path(a.points),
                 dcfg=DecisionConfig(relook_band=(a.rule_relook_low, DecisionConfig().relook_band[1])))
    print_report(report)
    print(f"  -> {Path(a.out_dir) / a.name / 'report.json'}")
    return report


if __name__ == "__main__":
    main()

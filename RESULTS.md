# Results log

Each row is one sweep. All runs are paired by seed (common random numbers). Hypotheses refer to
[PREREGISTRATION.md](PREREGISTRATION.md). That file was written at 17:38 UTC on 2026-10-26, before any
main run (SHA-256 `790902a3…`).

## Tier A v1 (calibrated) — main result

Perception is now `calibrated`: hiker detection probability and confidence come from the OpenCV pipeline
(YOLOX on TartanGround frames with composited cut-outs, `results/calibration.json`; relooks and scans use the
3× zoom table, `results/calibration_zoom.json`). The bear is found only out to about 15 m at native
resolution (20–30 m zoomed). Fog β (per metre, scattering model) limits hiker vision only; the predator's
sensing is not fogged. β = 0.05 ≈ 78 m visibility, β = 0.1 ≈ 39 m. Seeds 1000–1199, paired by seed.
Numbers come from `results/tierA_v1_report.json` (`uv run python -m lost_together.report results/tierA_v1.csv
--ref results/tierA_ref.csv --group results/tierA_v1_group.csv --control results/tierA_v1_off.csv`).

| Sweep | Episodes |
|---|---|
| `tierA_v1`: 8 speeds × 7 radii × solo/sightings/map/share × β 0/0.05/0.1 × 200 seeds | 134,400 |
| `tierA_v1_group`: group vs share, β = 0, same grid | 22,400 |
| `tierA_v1_off`: predator off, 5 modes × 3 β × 200 seeds | 3,000 |
| `tierA_ref` (from v0, perfect perception, β = 0): reused for H4 after a row-for-row reproducibility check with current code | 44,800 |

Grid-mean escape rate (fraction of the 4 hikers who reach an exit):

| β | solo | sightings | map | share | share, perfect perception |
|---|---|---|---|---|---|
| 0 | 0.268 | 0.327 | 0.322 | 0.424 | 0.547 |
| 0.05 | 0.270 | 0.323 | 0.324 | 0.419 | — |
| 0.1 | 0.251 | 0.285 | 0.305 | 0.355 | — |

| Hypothesis | Outcome (v1, calibrated) |
|---|---|
| H1 `share` > `solo` in most of the grid | **Holds at every fog level.** 56/56 cells have a CI above 0 at β = 0, 0.05 and 0.1. Mean Δ = +0.156 / +0.148 / +0.104. |
| H2 the `share` benefit shrinks with predator speed and sensing range | **Partly holds** (same as v0). It shrinks with speed (β = 0: +0.21 at 0.5× to +0.12 at 3×) but only slightly with sensing range (+0.17 at 50 m to +0.15 at 400 m). |
| H3 `sightings` helps in an intermediate band; predator-off control shows `sightings` ≈ `solo` | **Partly falsified** (same pattern as v0). The control holds: with no predator, `sightings` − `solo` = +0.025 [−0.020, +0.070] at β = 0, +0.015 [−0.029, +0.060] at 0.05, 0.000 at 0.1. With the predator, `sightings` helps in 41/40/38 of 56 cells (β = 0/0.05/0.1; at most 1 cell below 0). As predicted it does not help against a slow predator (0.5×: 0/7 cells above 0 at β = 0), but contrary to the prediction it **keeps helping when the predator senses the whole forest** (400 m: 6/8 cells above 0) and is strongest at 2–2.5×. |
| H4 imperfect vision costs escapes (`calibrated` < `perfect`) | **Holds, and the cost is larger than v0 assumed.** At β = 0, 208/224 mode × cell comparisons are below 0 and none above (mean −0.136 solo, −0.096 sightings, −0.175 map, −0.123 share). With fog: 211/224 (β = 0.05), 223/224 (β = 0.1), against the clear-air perfect reference, so these mix the fog and perception costs. |
| H5 `group` ≤ `share` | **Holds.** `group` is below `share` in 44/56 cells and above in none (mean −0.073; grid mean 0.350 vs 0.424). The penalty is largest against a slow predator (−0.16 at 0.5×, −0.03 at 3×). |

**Where does sharing stop helping as fog rises?** Not within the range tested. The `share` benefit falls by
a third from clear air to β = 0.1 (+0.156 → +0.104), but every cell stays above 0; the weakest is +0.039
[+0.016, +0.064] (3× speed, 200 m sensing, β = 0.1). Light fog (β = 0.05) changes almost nothing, because the
detector's ~15 m reach is already shorter than 78 m visibility; β = 0.1 starts to bite. `map` sharing is
nearly fog-proof (+0.054 at every β), so what fog erodes is the value of shared *sightings* (+0.059 → +0.034).

**Gap recovered by sharing:** (V_share − V_solo) / (P_share − V_solo) = **0.56** at β = 0 (0.54 at 0.05,
0.35 at 0.1). This is lower than v0's 0.81: with the measured detector, vision is the bottleneck, and sharing
recovers about half of what perfect perception would give.

**Predator-off control (β = 0):** solo 0.649, sightings 0.674, map 0.890, share 0.931, group 0.831. Most of
the navigation value of sharing is the map; sightings alone add nothing measurable without a threat.

Figures: `results/tierA_v1_headline_b{0,0.05,0.1}.png`; the β = 0.05 version is `docs/img/headline.png`
(`uv run python -m lost_together.figures results/tierA_v1.csv --control results/tierA_v1_off.csv --beta 0.05`).

## Tier C — live agent vs rule (Agentic Vision)

Both policies decide on the same 360 logged decision moments (seeds 2000–2059, `share`, 1.5× speed, 100 m
sensing, β 0/0.05/0.1), each rendered as a composited TartanGround frame pair: a bear at the true distance
for real sightings, and a deer, a dog or nothing at the claimed distance for false ones. The sample is
stratified at 168 real / 192 false; naturally only 18.3% of escalations are real, so rates are also
re-weighted to that mix. The agent is `qwen3-vl:30b-a3b-instruct` on Ollama (g6e.xlarge, L40S), with the
OpenCV tools `look_closer`, `compare_frames` and `query_map`, up to 3 tool turns. The rule gets the same
look-closer trigger (`--rule-relook-low 0.1`). Scoring: a decision is correct when it is `escape` exactly
when the evaluator says the hiker should escape (`regroup` counts as not escaping). Source:
`results/agent_bench/tierC_live_report.json` (`uv run python -m lost_together.agent.combine
results/agent_bench/tierC_live_a results/agent_bench/tierC_live_b`).

| | Rule | LLM agent |
|---|---|---|
| Accuracy (raw, n = 360) | 0.578 | 0.606 |
| Accuracy re-weighted to 18.3% real | 0.486 | 0.665 |
| False alarms (of 192 false) | 110 (57.3%) | 57 (29.7%) |
| Late escapes (of 168 real) | 42 (25.0%) | 85 (50.6%) |
| Per 100 natural moments: false alarms / late escapes | 46.8 / 4.6 | 24.3 / 9.3 |
| Fallback to rule | — | 0% |
| Decision latency p50 / p95 | 0.15 s / 0.16 s | 4.0 s / 8.0 s |
| $ per decision | 0 | $0.0026 (model time × $2.235/h); $0.0014 amortised over GPU wall time with two parallel streams |

- **McNemar, rule vs agent:** 49 points only the rule got right, 59 only the agent; exact p = 0.39. **No
  significant difference in accuracy.** The agent makes a different trade-off: it halves false alarms and
  doubles late escapes.
- **Fog is where the agent fails.** In this sample every β = 0.1 point is a real bear (66 points). The agent
  missed 46 of them, the rule 27. At β = 0 and 0.05 the agent is more accurate (0.66 vs 0.57 and 0.68 vs
  0.58, no per-stratum significance test), mostly by avoiding false alarms. Because the β = 0.1 stratum
  contains only real bears, fog is confounded with the real/false mix.
- **Scalar ablation** (100 points, same model given only the scalar features, no image and no tools;
  `results/agent_bench/tierC_ablation_report.json`): it answered `look_again` 92 times and never escaped, so
  0 false alarms and 42/42 late escapes. Its raw accuracy (0.58) is not significantly different from the
  full agent's (0.62; McNemar p = 0.66). Its re-weighted accuracy (0.82) equals the trivial "never escape"
  policy, which shows that re-weighted accuracy alone is a poor metric here: late escapes must be weighed
  separately.
- **Verdict.** The agent's visual tool use changes its decisions (it looks closer, compares frames, and
  calls off false alarms the rule acts on) and runs reliably (0% fallbacks over 360 points). It is not more
  accurate overall than the rule, and it is worse at catching real threats in thick fog. We do not claim
  that the agent is better.
- **Setup note.** The bare `qwen3-vl:30b` tag on Ollama 0.40.11 is the thinking variant and ignored
  `think: false` / `reasoning_effort: none`, giving empty answers (100% fallback on a 3-point smoke test); we
  used the instruct tag. GPU time for this session: 32 min (≈ $1.19, from `tools/gpubox.sh down`; not a billing record). The amortised $0.0014 figure uses the longer half's wall time, 833 s (`wall_s` in `tierC_live_b/report.json`).
- **Hero traces:** `viewer/data/agent/` (index.json): one correct escape after looking closer, one false
  alarm avoided (a deer), and one fog failure.

## COOL comparison — stock baseline only (COOL run blocked)

`tools/pipeline_bench.py` times the per-hiker-frame pipeline on 640×640 TartanGround frames with a bear
composited at 12 m in light fog (β = 0.05). Stages: render (composite + fog), detect (YOLOX-s on the new DNN
engine + dark blobs + haze), stereo (SGBM), plus the agent tools. Stock `opencv-python==5.0.0.93` on this
c8g.2xlarge (Graviton4, 8 vCPU, $0.21592/h on-demand in ap-south-1 per the AWS Pricing API). Its build info
(saved in `results/bench_stock_graviton.json`) already lists the **KleidiCV** HAL and NEON FP16/BF16/DOTPROD
dispatch, so a COOL speedup must be measured against this, not assumed.

| Median ms | render | YOLOX alone | detect | stereo | look_closer | compare_frames | per frame (render + detect + stereo) |
|---|---|---|---|---|---|---|---|
| 1 thread | 42.0 | 693.1 | 892.8 | 70.5 | 697.9 | 17.3 | 1,005 ms (0.99 frames/s) |
| 8 threads | 44.0 | 125.2 | 244.4 | 20.4 | 129.1 | 17.2 | 309 ms (3.24 frames/s) |

Cost per 1,000 frames: **$0.0185** measured with one 8-thread process. With 8 single-thread processes it
would be about $0.0076 (7.9 frames/s per box), but that is an extrapolation assuming linear scaling, not a
measurement. Detection is 79–89% of per-frame time (YOLOX alone 40–69%). The COOL AMI run is blocked: `run-instances --dry-run`
returns `OptInRequired` on the default account (Marketplace terms not yet accepted there).

## Tier A v0 — placeholder perception model (superseded by v1, calibrated)

The `surrogate` perception parameters here are guesses. They are not yet measured from the OpenCV
pipeline, so treat these numbers as a dress rehearsal of the analysis.

| Sweep | Episodes | Where |
|---|---|---|
| `tierA_v0`: 8 speed ratios × 7 radii × 5 cooperation modes × 200 seeds (1000–1199) | 56,000 | AWS c8g.2xlarge, about 15 min |
| `tierA_v0_off`: predator off, 5 modes × 200 seeds | 1,000 | AWS |
| `tierA_ref`: perfect perception, 4 modes, same grid and seeds | 44,800 | AWS |

| Hypothesis | Outcome (v0) |
|---|---|
| H1 `share` > `solo` in most of the grid | **Holds.** 56/56 cells have a CI above 0. Mean Δ = +0.159. |
| H2 the `share` benefit shrinks with predator speed and sensing range | **Partly holds.** It shrinks with speed (≈ +0.25 at 0.5× to ≈ +0.10 at 3×) but is roughly flat across sensing range. |
| H3 `sightings` alone helps in an intermediate band, and the predator-off control shows `sightings` ≈ `solo` | **Partly falsified.** The control holds: 0.612 vs 0.614 with no predator, so the benefit is about the threat and not about navigation. The benefit appears once the predator is *faster* than the hikers (speed ≥ 1.25×: +0.04 to +0.09, 32/56 cells above 0). Contrary to the prediction, it **persists when the predator senses the whole forest** (400 m), and is absent when the predator is slower than the hikers. |
| H4 imperfect vision costs escapes (`surrogate` < `perfect`) | **Holds.** 142/224 cells are below 0 and none are above. The mean gap is −0.049. |
| H5 `group` ≤ `share` | **Holds.** `group` is below `share` in 50/56 cells (mean −0.139). Groups are easier to sense and lose time gathering. |

**Gap recovered by sharing:** (V_share − V_solo) / (P_share − V_solo) = **0.81**. Sharing a map built from
imperfect vision recovers 81% of the gap between solo hikers with that vision and sharing hikers with
perfect perception.

Figure: `results/tierA_v0_headline.png` (regenerate with `uv run python -m lost_together.figures results/tierA_v0.csv --control results/tierA_v0_off.csv`).

## Robustness (v0 perception model, seeds 1000–1199)

Both sweeps use the 3 × 2 anchor grid: speed ratio 0.75 / 1.5 / 2.5 × sensing radius 20 / 40 cells.

| Sweep | Finding |
|---|---|
| `robust_ambush`: the predator waits near an exit | `share` − `solo` stays clearly positive (+0.12 to +0.28, all CIs above 0). `map` behaves the same way (+0.03 to +0.21). `sightings` − `solo` is above 0 only for a fast predator with a 40-cell radius (+0.05 / +0.06); elsewhere it is ≈ 0. |
| `robust_comm`: map sharing only within radio range | The benefit shrinks with range. At 50 m (10 cells) it is +0.05 to +0.10; at 100 m (20 cells) it is +0.08 to +0.13, against +0.12 to +0.27 with global sharing. All CIs are above 0. The value of a shared map scales with how far the hikers can talk. |

## Pilots (exploratory, seeds 0–119; used only to choose ranges)

- **`pilot1`:** `share` beat `solo` in 19/20 cells, and `group` ≈ `solo`.
- **`pilot2` and `control_off`:** most of the `share` benefit came from finding the exit (+0.30 with the predator off). This led to splitting sharing into `sightings` and `map`.
- **`pilot3` and `control_off2`:** first evidence that `sightings` helps only against a threat.

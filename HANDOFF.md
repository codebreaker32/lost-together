# Handoff: autonomous work on the AWS agent box

You are continuing "Lost Together", an entry to the OpenCV AI Competition 2026 (powered by AWS). Read
CLAUDE.md first (commands, rules, account split), then PROJECT_BRIEF.md §7 and §12 (official rubric and
award strategy), RESULTS.md and PREREGISTRATION.md. The goal is the highest score on the official rubric:
- **Overall:** technical execution 30, innovation 20, impact 20, UX 10, docs 10, cloud 10.
- **Agentic Vision award** and **Best Use of COOL award:** each scored separately.

Work in priority order. After every meaningful step, append a dated line to **PROGRESS.md**: what you did,
the key numbers, and what's next. The user checks that file.

## Where things stand at handoff (2026-10-26 ~20:15 UTC)

- **Simulator, experiment and pre-registration:** done. The pre-registration is committed (`05e5a12`, branch `dev`) and is immutable.
- **Tier A v0** (placeholder perception): done and written up in RESULTS.md.
- **Vision pipeline:** done. Calibration tables are in `results/calibration.json` and `results/calibration_zoom.json` (3× zoom).
  - Key fact: YOLOX finds the bear only out to about 15 m on these frames. The 3× "look closer" zoom reaches about 20–30 m.
  - The zoom table is an upper bound, by up to about 0.08, because the zoom tile uses the true ground-contact row.
- **Calibrated perception in the simulator:** `Perception.CALIBRATED`.
  - Relooks use the zoom table.
  - A teammate's report makes a hiker scan with zoom; see `wants_scan` and `Perceiver.scan`.
- **Tier A v1** (calibrated: 8 speeds × 7 radii × solo/sightings/map/share × β 0/0.05/0.1 × 200 seeds): it ran on THIS box, and the output is in `results/tierA_v1.csv`. **Not run yet:**
  - `tierA_v1_group`: group vs share at β = 0.
  - `tierA_v1_off`: the calibrated predator-off control.
- **Hero episode:** seed 45, calibrated, β = 0.05. Solo gets 2/4 out; the shared map gets 4/4.
  - Camera views come from `lost_together/camera_tiles.py`. Each view records `sim_detected` and `render_agrees`.
  - Exported to `viewer/data`.
- **Viewer redesign** (`/impeccable` + taste-skill-v2): it was running in the user's local session at handoff. The local session will sync the finished `viewer/` here. **Do not edit `viewer/` until PROGRESS.md says the redesign has landed.**
- **Tier C:** 360 stratified decision moments (168 real + 192 false; the natural real fraction is 0.183).
  - Frames are rendered into `results/tierC/` (points.jsonl + frames/).
  - The bench is `lost_together/agent/bench.py`. `--rule-relook-low 0.1` gives the rule the same look-closer trigger as the agent.
- **Hosting:** live at https://6ur5dkmovrz7eqll6dehfjgg7a0vmzog.lambda-url.ap-south-1.on.aws/, but it returned 403 until the user reran `infra/aws/deploy_lambda_site.sh` with the permission fix. Making things public is the user's call; don't change public-access settings yourself. CloudFront needs AWS Support account verification, which the user has requested.
- **COOL:** the user subscribed, but `run-instances --dry-run` with the COOL AMI (`ami-019a34dc69ce62cb0`, ap-south-1) still said `OptInRequired` at handoff. Re-check before planning any COOL work.

## Plan (do in order; skip a step only if blocked, and say why in PROGRESS.md)

1. **Finish Tier A v1** on this box (the box has 8 vCPUs; use `--workers 7`):
   ```bash
   uv run python -m lost_together.sweep --name tierA_v1_group --perception calibrated --beta 0 --speed-ratio 0.5 0.75 1 1.25 1.5 2 2.5 3 --sense-radius 10 15 20 30 40 60 80 --coop group share --seeds 200 --seed-offset 1000 --workers 7
   uv run python -m lost_together.sweep --name tierA_v1_off --perception calibrated --beta 0 0.05 0.1 --predator-on false --coop solo sightings map share group --seeds 200 --seed-offset 1000 --workers 7
   ```
2. **Analyse v1** against H1–H5:
   - Use the same method as the v0 section of RESULTS.md: paired bootstrap per cell, counts of significant cells, and gap recovered.
   - Facet by β. The new question is where sharing stops helping as fog rises.
   - Add a "Tier A v1 (calibrated) — main result" section above v0 in RESULTS.md, with honest verdicts.
   - Regenerate the headline figure per β with `lost_together/figures.py`, and copy the β = 0.05 version to `docs/img/headline.png`.
   - Write `viewer/data/results.json` as `{headline: [{label, value, note}], figure: "results-headline.png"}`, and copy the figure next to it. Real numbers only. This one file in viewer/ is allowed.
3. **Tier C live** (Agentic Vision award). This step costs GPU money: $10 hard cap, about $1.40 spent so far.
   - Stub smoke test first: `uv run python -m lost_together.agent.bench results/tierC/points.jsonl --mode stub --name tierC_stub --limit 10 --rule-relook-low 0.1`.
   - Then:
     ```bash
     tools/gpubox.sh up && tools/gpubox.sh tunnel
     # split points into 2 files and run both halves at once (Ollama serves parallel requests):
     uv run python -m lost_together.agent.bench results/tierC/points_a.jsonl --mode live --base-url http://localhost:11434/v1 --model qwen3-vl:30b --name tierC_live_a --rule-relook-low 0.1
     ```
   - Then a `--ablation scalar` run on about 100 points.
   - **Then IMMEDIATELY `tools/gpubox.sh down`.** Check `tools/gpubox.sh status` for spend.
   - Report accuracy, false alarms, late escapes and McNemar, both raw and re-weighted to the natural real fraction of 0.183. Also report p50/p95 latency and $ per decision.
   - Save 2–3 agent decision traces for hero moments into `viewer/data/agent/` (that's allowed).
4. **COOL** (Best Use of COOL award), only if the dry run succeeds:
   - Launch the COOL AMI on Graviton, inside the 8-vCPU quota. This box already uses 8, so do it after the sweeps by stopping work here, or ask the user.
   - Benchmark the per-frame perception pipeline (YOLOX, SGBM, composite) in three setups: stock pip OpenCV on Graviton vs COOL on Graviton, and stock on x86 if quota allows.
   - Log `cv2.getBuildInformation()` as proof, and report frames/s and $ per 1,000 frames.
   - Terminate when done.
5. **Docs:** fill every `{{TBD}}` in README.md, docs/WRITEUP.md and docs/VIDEO_SCRIPT.md with the v1 and Tier C numbers. Then run one skeptical-judge review pass: every number traced to a results file, nothing invented.
6. **Redeploy the viewer** after the redesign lands: `bash infra/aws/deploy_lambda_site.sh`. If it needs the public-permission step and the auto-mode classifier blocks it, leave it for the user and note it in PROGRESS.md.

## Rules (from the user; non-negotiable)

- **Accounts:**
  - This box and the sweeps run on the default account (…8661) via profile `lost-together-main`.
  - The GPU runs only on account …0916 via profile `lost-together`.
  - Never use or create root credentials.
- **Billing:** never leave billable resources idle. The GPU box goes down right after each session; check `tools/gpubox.sh status`. Before ending a work block, audit running instances in both accounts.
- **Other people's resources:** don't touch resources the team didn't create through this project (anything without `project=lost-together`).
- **Commits:** only when the user asks in chat. The pre-registration is never edited.
- **Ground truth:** no ground-truth leakage into hiker-side code. Keep `test_hiker_policy_never_touches_ground_truth` passing, and run `uv run pytest -q` before you finish anything.
- **Honesty:** report what holds and what doesn't. Calm wording ("intercepted", never gore). Credit every dataset and cut-out licence.
- **Secrets:** `.env` holds `GROQ_KEY`. Never print secrets or commit `.env`.

## Update from the local session (2026-10-26 20:40 UTC): local work is wrapped up
- The user's local session has stopped. **This box is now the only working copy**; nothing more will be synced from local.
- **Viewer redesign:** the first design pass is now in `viewer/` (plus PRODUCT.md), but its finish review and fix were never run. Follow **docs/VIEWER_BRIEF.md** to finish it, then redeploy with `bash infra/aws/deploy_lambda_site.sh`. This supersedes the earlier "do not edit viewer/" note.
- **Hero camera views:** `viewer/data/frames/` now holds the final renders, at the predator's true position, with sim_detected / render_agrees (25/38 real-predator views agree; all 16 "not detected" views agree).
- **COOL:** still blocked. The user must accept the Marketplace terms on the default account …8661.

## Hard stop: this box stops itself at 04:30 UTC on 2026-10-27 (10:00 IST). This is the user's instruction.
- A shutdown is scheduled (`shutdown -h 04:30`), and the shutdown behaviour is *stop*, so the disk and files are kept. **Never cancel it.**
- Plan to finish by **04:00 UTC**. By then the GPU box must be down (`tools/gpubox.sh down`), the docs filled in, the viewer finished and redeployed, `uv run pytest -q` green, and PROGRESS.md carrying a final "state at stop" summary.

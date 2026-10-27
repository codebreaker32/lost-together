# Viewer redesign: brief and status

**Status (handed over 2026-10-26 20:35 UTC):**
- **Done:** the design-and-build pass ran in the local session. It wrote new `viewer/index.html`, `app.js`, `style.css` and `fonts/`, plus `chev-*.svg`, `favicon.svg` and `PRODUCT.md`.
- **Done (2026-10-27):** finish review and fix pass (impeccable + taste-skill-v2); DESIGN.md written. Hero camera frames re-rendered with the run's fog (β 0.05).
- **To finish it:**
  1. Load the `impeccable` skill (installed on this box) and read `~/.agents/skills/design-taste-frontend/SKILL.md` (the team's taste-skill-v2).
  2. Run a critique / finish review of the current viewer against this brief. Take headless-Chromium screenshots at 1920×1080 and 390 px.
  3. Fix everything in one batch and re-verify once.
  4. Then redeploy with `bash infra/aws/deploy_lambda_site.sh`.

## Brief

The team lead said the old UI "looks complete AI slop" and made the project look like "only a simulation".
Judges spend about 5 minutes per entry, and this page is also the backdrop of the 2-minute demo video. The URL params used in `docs/VIDEO_SCRIPT.md` must keep working: `left`, `right`, `t`, `speed`, `play`, `eval`, `risk`, `reveal`, `ff`, `capture`, `hiker`.

**Product truth the page must communicate:**
- **It's a computer-vision system.**
  - Each hiker sees the forest through a camera, and OpenCV 5 (YOLOX on cv2.dnn ENGINE_NEW) detects the predator. Detection only works out to about 15 m on a 640 px frame.
  - "Look closer" re-runs the detector on a **3× zoom tile** and confirms out to about 20–30 m.
  - Hikers share sightings in one living map, and a teammate's report makes a hiker scan with zoom.
  - A rule-based planner and an LLM agent (qwen3-vl on an AWS GPU, tools via an OpenCV 5 MCP server) decide, with a human "ranger" approval step.
- **The hero view** is SOLO vs SHARED MAP for the same seed: seed 45, calibrated perception, fog visibility 78 m. Solo gets 2 of 4 out; 2 are intercepted without ever seeing the predator. The shared map gets 4 of 4 out. Use calm wording ("intercepted", "got out").
- **Camera views come first.** For the followed hiker, show:
  - the real camera frame, `viewer/data/frames/<episode_id>/<t>_<hiker>.jpg`, with OpenCV detection boxes drawn by the viewer from `frames/<episode_id>/manifest.json`;
  - the 3× look-closer tile, `frames/<episode_id>/zoom/<t>_<hiker>.jpg`, with its detections, when present.
- **Manifest schema:**
  ```
  {episode, beta, model, views: [{t, hiker, kind: "real"|"decoy"|"quiet", distance_m, background,
   truth_box [x,y,w,h], detections: [{cls, score, box [x,y,w,h] (640×640 px), matches_truth}], attempt,
   zoom?: {factor, origin, truth_box, detections}, sim_detected?, render_agrees?}]}
  ```
- **Show the evidence honestly:**
  - In the evaluator view, label detections that don't match the truth as false positives.
  - `render_agrees = false` means this sample render disagrees with the simulator's calibrated draw. Show a small honest note; never hide it.
- **Data sources:**
  - **Episode JSON:** `viewer/data/<id>.json` from `lost_together/export.py`. It holds the map, frames (t, h, p, s, e with "relook"/"scan"/"escape"/"home" events), metrics and config. `index.json` lists episodes and camera tiles.
  - **Agent traces:** schema from `lost_together/agent/loop.py` `write_trace`, loaded from `viewer/data/agent/*.json` when present. The panel must look intentional when they're absent; no fake data.
  - **Experiment results strip:** `viewer/data/results.json` as `{headline: [{label, value, note}], figure}`. Real numbers only.
- **Constraints:**
  - Colour-blind-safe palette and full keyboard control.
  - Quality good enough to film at 1920×1080, plus a usable phone layout.
  - **No external requests:** CSP is `default-src 'self'`, so no CDNs and no internet web fonts. Use only vendored font files with their licence.
  - Fast load.

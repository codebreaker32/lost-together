# Product

<!-- impeccable:product-schema 1 -->

> Written by a workflow subagent with no channel to the team for the init interview. Facts below come from
> PROJECT_BRIEF.md, README.md, CLAUDE.md, docs/VIDEO_SCRIPT.md and the team lead's redesign brief.
> Lines marked *(inferred)* are assumptions the team has not confirmed.

## Platform

web

## Users

- **Competition judges** (OpenCV AI Competition 2026, powered by AWS). About five minutes per entry, usually on a
  laptop or desktop browser, sometimes a phone. They score technical execution (depth of the OpenCV 5
  implementation and its evaluation) highest, then innovation, impact, UX, documentation and cloud delivery.
  Agentic-award judges also score failure handling and human control.
- **The video audience.** The replay viewer is the backdrop of the 2-minute demo video, screen-recorded at
  1920×1080 with burned-in captions.
- **The team** uses it to check exported episodes before recording. *(inferred)*

## Product Purpose

Lost Together is a pre-registered simulation study. Simulated hikers lost in a forest each see the world only
through their own camera. OpenCV 5 (YOLOX on `cv2.dnn` ENGINE_NEW) detects a predator in those frames, the hikers
share what they see in one living map, and a rule-based planner plus an LLM agent decide when to escape, with a
human ranger approving evacuation orders. The research question is whether sharing helps, and where it stops
helping.

The replay viewer's job is to make a judge see, within a minute, that this is a computer-vision system whose
camera evidence changes what the hikers do, and that sharing the map changes the outcome for the same forest.

## Positioning

The decisions in this simulation are driven by real detector output on real-looking frames (TartanGround forest
frames with composited animals and physical fog), not by a coin flip. The viewer shows the actual frame, the
actual YOLOX boxes and scores, the 3× look-closer re-run, and the honest disagreements between a sample render
and the simulator's calibrated draw.

## Operating Context

- Static files served by `python -m http.server -d viewer 8000` locally, later from S3 + CloudFront with CSP
  `default-src 'self'`. No external requests of any kind.
- Data comes from `lost_together/export.py` (episodes, index.json), `lost_together/camera_tiles.py`
  (frames/<episode>/*.jpg + manifest.json) and, later, agent traces from `lost_together/agent/loop.py`.
- URL parameters drive the video shots: `left, right, t, speed, play, eval, risk, reveal, ff, capture, hiker`.

## Capabilities and Constraints

- Side-by-side replay of two cooperation conditions for the same seed (common random numbers).
- Hero contrast: seed 45, calibrated perception, fog β = 0.05 (meteorological visibility ≈ 78 m). SOLO gets
  2 of 4 hikers out (2 intercepted, never saw the predator); SHARED MAP gets 4 of 4 out.
- Detector range: about 15 m on a 640 px frame; the 3× look-closer tile confirms out to about 20–30 m.
- A teammate's report makes a hiker scan with zoom (the `scan` event).
- Evaluator view: only the evaluator knows which sightings were real. Hikers and the agent never see ground truth.
- Agent traces and an experiment-results summary are optional inputs; nothing may be shown when they are absent.
- Wording is calm and non-violent: "intercepted", "got out". Never "killed", "attacked".

## Brand Commitments

- Name: **Lost Together**.
- Colour-blind-safe palette (the team has used Okabe–Ito).
- Everything shown is labelled as simulated; stub-model agent output is never presented as a result.

## Evidence on Hand

- `viewer/data/*.json` episodes for seed 45 (solo, sightings, map, share), `viewer/data/index.json`.
- `viewer/data/frames/<episode>/` camera frames, zoom tiles and manifest.json (being re-rendered).
- `docs/img/headline.png`, `fog_strip.jpg`, `stereo_depth.jpg`.
- RESULTS.md (Tier A v0, guessed perception; v1 calibrated numbers are TBD).
- No live LLM results yet; only stub traces in `results/agent_bench/synth_stub/` (not a result).

## Product Principles

1. Show the evidence, then the decision. A camera frame with its detections comes before any claim.
2. Honest by construction: false positives, render disagreements and stub runs are labelled, never hidden.
3. Same forest, twice. Every comparison is paired by seed and says so.
4. Calm. The subject is danger; the presentation is a field instrument, not a thriller.

## Accessibility & Inclusion

- Colour-blind-safe palette; state never carried by colour alone (shape and text too).
- Full keyboard control; visible focus.
- Legible when filmed at 1920×1080 and usable on a ~390 px phone.

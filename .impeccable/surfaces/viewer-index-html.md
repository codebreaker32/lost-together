---
version: 1
slug: "viewer-index-html"
primary_target: "viewer/index.html"
related_targets: ["viewer/app.js","viewer/style.css"]
---

# Surface: replay viewer (viewer/index.html)

Mode: Operate (a replay instrument judges and the video audience read), with a Read tail below the fold.
Audience: competition judges (~5 min), the 2-minute video (1920x1080 capture), the team checking exports.
Task: watch the same forest run twice (SOLO vs SHARED MAP), follow one hiker's camera, see OpenCV detections and
look-closer re-runs drive decisions, inspect the evaluator truth, read agent traces and findings when present.
Constraints: static, CSP default-src 'self', no external requests, URL params left/right/t/speed/play/eval/risk/
reveal/ff/capture/hiker keep working, colour-blind-safe, full keyboard, phone layout.
Unresolved: agent traces and results.json do not exist yet; the copy for those sections is empty-state only.
Decision round: probed once via the decision page (no answer in 60 s, workflow subagent); proceeded with the roll.

## Direction contract

THESIS: A search-and-rescue incident board for one forest run twice: each hiker's real camera frame and its OpenCV
detections lead, the map that evidence builds sits above it, the outcome line heads each run. Refuses the dark
sci-fi simulation dashboard with a sidebar of rounded cards.

OWN-WORLD: Cool laminated-white ground, ink-green black, a solid black title band (ICS form / NPS Unigrid band).
National Park (OFL, routed trail-sign lettering) as the one family, tabular figures. Okabe-Ito colours with fixed
jobs: vermilion = predator only, amber = a report/sighting, blue = got out, map greens from orienteering maps.
Square corners, tone steps between surfaces, rules only in tables and the map frame.

STORY: The visitor sees two outcomes for one seed, presses play, watches hikers walk, sees what a camera saw and
what YOLOX found (and what it got wrong), sees a teammate's report trigger a scan, and leaves knowing the shared
map changed the outcome. Raises: claim-and-proof pairing (monochrome canon); one colour one job (terminal yellow);
focused hiker grows, others dim (title wall); tabular figures everywhere (datamatics).

FIRST VIEWPORT: 1920x1080. Black band (wordmark, run label "Simulation · seed 45 · calibrated perception · fog
visibility 78 m", run pickers, help). Outcome line over each map at 40px. Two 576px square maps side by side, each
with a 576x288 camera strip beneath (frame + 3x look-closer tile). Right column: hiker roster, field log. Layer
toolbar above the maps; transport bar at the bottom with event ticks.

FORM: Search-and-Rescue Incident Board, position 5 of my ordered list; seed key 4e41ceb1. Signature move: the
look-closer loupe, a leader drawn from the zoom region on the camera frame to the 3x tile where the detector re-ran.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

# Video script: Lost Together (2:00)

Follows PROJECT_BRIEF §11. Screen-recorded, with burned-in captions that match the voice-over word for word. The tone is calm and factual. Use "intercepted", never "killed" or "attacked". No gore, and no attack animation: a caught hiker simply leaves the map. Every number spoken must match [RESULTS.md](../RESULTS.md) at recording time, and the `{{TBD}}` values are filled in from the calibrated v1 rerun. Run label on screen throughout: "Simulation · Tier A v1 (calibrated)", or "v0 dress rehearsal" if v1 is not ready.

**Before recording**

```bash
uv run python -m lost_together.export --find-contrast 60 --coops solo sightings map share group --left solo --right share --out viewer/data
python -m http.server -d viewer 8000
uv run python -m lost_together.vision.demo          # refreshes data/demo/*.png
```

- Browser at 1920×1080, zoom 100%, nothing else on screen.
- In the viewer, `capture=1` hides the controls (the `C` key toggles it).
- The hero seed is 45 (calibrated perception, light fog β = 0.05, speed ratio 1, sensing radius 20): SOLO gets 2 of 4 hikers out (2 intercepted), SHARED MAP gets 4 of 4. It was picked by `--find-contrast` as a clear example and lies outside the main seeds (1000–1199), so the voice-over must not present it as typical (the grid average at β = 0.05 is +15 points). The older seed-26 files in `viewer/data` use the v0 surrogate perception; do not show them under the v1 label.

---

## 0:00–0:20 · The problem

| Time | Shot (exact) | Voice-over |
|---|---|---|
| 0:00–0:06 | Full screen `docs/img/fog_strip.jpg`, top-left tile only (bear at 10 m, clear, "bear 0.82"). Slow zoom toward the box. | "Four hikers are lost in a forest. Each one sees only what their own camera sees." |
| 0:06–0:13 | Same image: pan across the top row, from clear to β = 0.05 ("bear 0.71") to β = 0.1 ("elephant 0.35"). | "Somewhere in the fog there is a predator. As the fog thickens, OpenCV's detector grows less sure." |
| 0:13–0:20 | `docs/img/stereo_depth.jpg`: left frame, ground-truth depth, StereoSGBM depth. Caption: "OpenCV 5: YOLOX on the new DNN engine · StereoSGBM · RGB-D odometry". | "The same question applies to any group caught by a moving hazard: wildfire, rising water, a bear. Does sharing what each of them sees help them get out?" |

## 0:20–1:00 · Same forest, twice

| Time | Shot (exact) | Voice-over |
|---|---|---|
| 0:20–0:27 | Viewer: `http://localhost:8000/?left=seed45-solo-sr1-R20-cal-b0.05&right=seed45-share-sr1-R20-cal-b0.05&hiker=4&reveal=end&capture=1&play=1&speed=20`. The left map is SOLO and the right is SHARED MAP. Same seed: same forest, same start, same predator. | "Here is the same forest, run twice with the same random seed. On the left, every hiker keeps a private map. On the right, they share one." |
| 0:27–0:40 | Keep playing. With `hiker=4`, the left pane shows hiker 4's own map. On the right, the shared map fills in faster. Press `R` to show the risk heatmap. | "On the right, every sighting goes onto the shared map. Old evidence fades, so the map stays current. That is what we call a living map." |
| 0:40–0:52 | Let the episode run to the end (`reveal=end` hides the outcome until then). The left shows 2 of 4 out and 2 intercepted. The right shows 4 of 4 out. | "In this forest, alone, two hikers reach an exit and two are intercepted. Sharing, all four get out. This is a clear example, not the average." |
| 0:52–1:00 | Press `E` (evaluator view): real sightings in vermilion, false ones in grey. Caption: "Only the evaluator knows which sightings were real." | "Only a separate evaluator knows which sightings were real. The hikers and the agent never see the ground truth." |

Camera views are exported for the hero pair (`viewer/data/frames/`): the viewer shows the followed hiker's full frame and the 3× look-closer tile above each map, with the real YOLOX boxes.

## 1:00–1:30 · The agent looks again, and a person decides

| Time | Shot (exact) | Voice-over |
|---|---|---|
| 1:00–1:12 | A decision trace from the **live** Tier C run, `viewer/data/agent/s2035_t20_h1_b0.05_llm.json` (point s2035_t20_h1_b0.05, light fog), shown as a scrolling card. Sequence from the trace: observe (two "unknown" blobs at 0.43 and 0.41, and a faint YOLOX "bear" #2 at 0.16) → `query_map` (risk 1.86) → `compare_frames` on #2 (moving, 1.0 px) → `look_closer` on #2 (defog + CLAHE: "bear" 0.16 → 0.58) → escape. On screen: "faint bear 0.16 → looked closer → bear 0.58 → escape". Truth: a real bear. | "When a sighting is uncertain, the hiker's AI agent can look again. It calls OpenCV tools to zoom and defog, to check whether the shape moved, and to read the shared map. Then it decides." |
| 1:12–1:22 | Terminal: `uv run python -m lost_together.agent.loop <points.jsonl> --index <i> --mode replay --autonomy advise --approver cli`. The ranger console shows `=== RANGER CONSOLE: pending ESCAPE order for <point> ===` with the agent's story and its reason, then `[a]pprove  [r]eject -> regroup  or type an action`. Type `a`. | "An escape order is an evacuation broadcast, so a ranger approves it or overrides it. Every approval is logged." |
| 1:22–1:30 | Split screen. Left: the agent's tool list from the MCP server (`detect_candidates`, `look_closer`, `compare_frames`, `query_map`). Right: the fallback line from a trace, "fallback_reason: …". | "If the model times out or gives an invalid answer, the hiker falls back to the rule-based planner. It never acts on a broken answer." |

The live run is done, so use the live trace above. Because the shot shows a success, keep this caption on screen from 1:00 to 1:12 so the clip does not overclaim: **"On 360 test moments the agent was not more accurate than the rule (McNemar p = 0.39): fewer false alarms, more missed bears in thick fog."**

If the live run is not ready, use the stub trace and keep this caption on screen the whole time: **"Pipeline demo with a stub model — not a result."** Do not speak any numbers from it.

## 1:30–2:00 · Where sharing helps

| Time | Shot (exact) | Voice-over |
|---|---|---|
| 1:30–1:42 | `docs/img/headline.png` (v1 version), left panel "Share everything". Hold, then highlight the column at speed 3×. | "Across 56 combinations of predator speed and sensing range, 200 paired forests each, sharing raised the share of hikers who escaped from 27% to 42%. The benefit shrinks as the predator gets faster. In thicker fog it shrinks by a third, but in every setting we tested, sharing still helped." |
| 1:42–1:52 | Same figure, middle panel "Share predator sightings only", then the right-hand control panel. | "Sharing sightings alone helps only once the predator is at least as fast as the hikers. We had predicted it would stop helping when the predator can sense the whole forest. It didn't, and we report that." |
| 1:52–2:00 | End card: project name, "OpenCV 5 · AWS Graviton + GPU · pre-registered · all simulated", repo URL {{TBD}}, viewer URL {{TBD}}. Credits line: "Forest frames: TartanGround (CC BY 4.0). Animal and hiker photos: public domain, CC0, CC BY 2.0 (see README)." | "Lost Together. Everything here is simulated, and every prediction is reported, right or wrong." |

---

**Caption and safety checklist**

- [ ] No word "kill", "attack", "maul" or "death" in the voice-over or captions.
- [ ] Every spoken number matches RESULTS.md, with the tier and perception model labelled on screen.
- [ ] The stub model is never presented as a result.
- [ ] Credits on the end card match the README's licence table (TartanGround CC BY 4.0; `bear_black_cdfw` CC BY 2.0 if shown).
- [ ] No real people on screen other than the public-domain hiker photos used as cut-outs.

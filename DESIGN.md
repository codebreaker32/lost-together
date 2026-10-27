# Design: Lost Together replay viewer

Derived from the shipped `viewer/` (index.html, style.css, app.js) on 2026-10-27 (refined the same day: type scale, hairlines, camera labels, map plate). The brief and status are in
docs/VIEWER_BRIEF.md and the product truth is in PRODUCT.md. If this file and the code disagree, the code wins.

## World

A search-and-rescue incident board: a laminated off-white ground, a solid black title band, ink-green black text
and square corners. It should read as a field instrument, not a sci-fi dashboard. The subject is danger, and the
presentation stays calm.

## Type

- One family: **National Park** (SIL OFL, vendored in `viewer/fonts/`, licence at `fonts/OFL.txt`) in 400 and 700.
  Tabular figures everywhere (`font-variant-numeric: tabular-nums`).
- Monospace is used only for literal code (`cv2.dnn`, file paths).
- A fixed role scale, as tokens in `style.css`: `--t-xs` 12 (tick and step counters, tags), `--t-sm` 13 (table
  heads, badges, chips, legend), `--t-md` 14 (captions, secondary lines, the log), `--t-base` 15 (body), `--t-lg` 17
  (run names, side headings), `--t-xl` 20 (clock, decision columns), `--t-2xl` 26 (wordmark, verdict, decision
  heads), `--t-3xl` 32 (section heads below the board), `--t-display` 38 (the outcome line and result values).
  The outcome drops to 32px under 1600px and 28px on phones.
- Canvas labels (detections, tile titles, map tags) are 12–13px, with the class name in 700 and the score in 400.
- Display sizes are tracked tight (-0.025em); run names in capitals get +0.03em.

## Colour (Okabe–Ito; every colour has one job)

| Token | Value | Job |
|---|---|---|
| `--predator` | #d55e00 vermilion | the predator, and a detection the evaluator confirms is the predator. Nothing else. |
| `--report` | #e69f00 amber | a report, sighting or detection as the hiker sees it |
| `--truth` | #f0e442 yellow | evaluator ground truth only (truth corners, the evaluator badge and toggle) |
| `--h0..h3` | blue, reddish purple, bluish green, sky blue | hiker identity |
| `--ink`, `--band` | #131a16, #101612 | text, title band, verdict |
| `--ground`, `--paper`, `--step`, `--step-2` | tone steps | surfaces are separated by tone first |
| `--rule`, `--rule-2` | #c3cac0, #d5dbd2 | 1px hairlines: controls and table heads, then rows |

State is never carried by colour alone. Got out is a check, intercepted is a dashed ring, escaping is a solid ring,
heading home is a dashed ring, a false positive is a dashed white box with words, and a false cue is a crossed ring.

## Spacing and edges

- A 4px spacing scale (`--s1` 4 … `--s10` 96). Groups are tight (4–12px), sections are generous (48–96px below the
  board).
- Square corners everywhere. Controls carry 1px borders; the one heavy rule is `--heavy` (2px ink), on the tops of
  result columns, pipeline steps, trace steps and the colophon.
- Control heights: 32px in the title band, 40px in the transport.

## Layout

- **1080p and taller (`body.fit`):** one viewport. Band, then two run columns and a side column, then the transport.
  - Each run column, in order: the run name and outcome; the **camera strip** (full frame plus the 3× look-closer
    tile, each a square at half the column width); a caption of up to three lines; the map.
  - The map takes the height that is left over. The map plate is exactly `--M` square and shares the strip's left
    edge; its canvas carries a transparent `--pad` margin that overhangs into the gutter for exit marks. The room
    beside it becomes a rail holding the map badges and that run's latest events.
  - The side column holds the paired verdict, the hiker roster (choose which camera to follow), a compact map legend
    (evaluator marks replace the report mark in the evaluator view) and the field log.
- **Shorter desktop screens:** same columns. The map is as wide as the strip and the page scrolls, so camera
  frames never shrink to thumbnails. The transport bar is sticky.
- **Under 1100px:** two columns, then one under 720px. The side column drops below the runs, and the layers sit
  behind a "Layers" button.
- **Below the board:** the experiment results (numbers in a row of columns at least 250px wide, values at up to 38px
  so the longest value fits; the figure full width), agent decisions, how it works, the legend and keyboard, and the
  colophon.

## Map rendering

- The plate sits on the board ground inside a 1px ink neatline.
- Routes: other hikers as fine lines (1.25px+, 60% opacity); the followed hiker on top at 2px+ over a paper casing.
- Exit labels: a known exit gets an ink tag with paper letters; an unseen exit gets a paper tag with a hairline.
- Map tags and labels snap to whole pixels.

## Signature: the camera strip

The followed hiker's real frame with OpenCV detection boxes drawn by the viewer. The 3× tile is joined to it by
leader lines from the zoom region (the loupe). Tile titles are placed first, and every later label steps around
them, so labels never overlap.

- Detection boxes snap to whole pixels: a 2px stroke over a 4px dark casing, so the edge holds on sky and foliage.
- The label tab sits flush on the box's top edge, aligned to its outer left edge (below the box when there is no
  room above).
- Truth corners sit 3px outside the box on the same casing. The loupe is a 1px white square on a casing, with 1px
  leaders at 60% white.
- Both tile titles use the same dark translucent tab.

## Honesty rules the UI enforces

- In the evaluator view each box is labelled as the predator, a decoy, or a false positive. The caption counts
  true detections, detections on the decoy and false positives separately.
- When `render_agrees` is false the caption says "Differs from the run:", in words that depend on `sim_detected`.
- When `manifest.beta` differs from the run's β, the frame carries a label such as "Rendered without fog · run used β 0.05".
- In agent traces, an `auto` ranger step reads "approved automatically: batch run, no person reviewed it".
  A model or source containing "stub" raises a "not a result" banner.
- Optional inputs that are absent (results.json, agent traces) show an empty state that explains what will appear.
  They never show placeholder numbers.

## Constraints

- CSP `default-src 'self'` (also set as a meta tag). No external requests and no inline style attributes; the
  viewer writes styles only through the CSSOM.
- URL params `left right t speed play eval risk reveal ff capture hiker` (plus `ring trail map`).
- Full keyboard control (`?` lists the keys) and a visible 3px focus ring.
- Shadows are only on things that float: toasts over the map, the dialog, the sticky transport. Hairlines are on
  controls, tabular lists (roster, field log, trace steps) and the transport's top edge.
- Motion has three uses: colour transitions on controls (140ms), a new event sliding into the rail during playback
  (280ms, ease-out), and the field-log highlight. `prefers-reduced-motion` removes all three.
- The scrubber is a 4px track filled to the playhead, with a 6px square playhead. Event ticks sit above (L) and
  below (R) the track.

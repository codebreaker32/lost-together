# Pre-registration — Lost Together main sweeps

Written 2026-10-26, before any main (Tier A/B/C) run. Pilot sweeps `pilot1`–`pilot3`, `control_off` and
`control_off2` (seeds 0–119) were exploratory: they were used to choose parameter ranges and to split
sharing into its two components. Main runs use fresh seeds starting at 1000. Every prediction below is
reported whether it holds or fails.

## Design

- **Simulator:** L0 grid world (`lost_together/`). N = 4 hikers start together, enjoy the forest, then
  leave on alarm or at `enjoy_steps`. The predator pursues the nearest hiker it senses. Capture removes
  that hiker and the predator then pauses (`handling_steps`).
- **Cooperation conditions:**
  - `solo` — each hiker keeps a private map.
  - `sightings` — only predator sightings are shared.
  - `map` — only terrain and exits are shared.
  - `share` — everything is shared.
  - `group` — share, plus moving together after a majority vote.
- **Perception:**
  - `surrogate` — the noise model; the main figure uses it after calibration from the OpenCV pipeline.
  - `perfect` — the reference.
- **Axes (phase diagram):** predator speed ÷ hiker speed ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3}
  × sensing radius (cells) ∈ {10, 15, 20, 30, 40, 60, 80}.
- **Fixed:** everything else at `config.py` defaults.
- **Seeds:** 200 per cell (seeds 1000–1199), identical across conditions (common random numbers).
- **Primary outcome:** fraction of hikers reaching an exit within T_max, per episode.
- **Primary contrasts:** `share − solo` and `sightings − solo`, paired by seed. Each is reported with a
  95% bootstrap CI per cell; cells whose CI includes 0 are masked.
- **Control:** predator off, all conditions, 200 seeds.

## Predictions

| ID | Prediction | Falsified if |
|---|---|---|
| H1 | `share` beats `solo` in most of the grid. | The CI includes 0 in a majority of cells. |
| H2 | The `share − solo` benefit shrinks as the predator gets faster and senses further. | The benefit is flat or grows along both axes. |
| H3 | Sharing **sightings** alone helps against the predator in an intermediate band (speed ratio ≈ 1–2, radius ≈ 15–40). It does not help when the predator is slow, or when it senses the whole forest. With the predator off, `sightings` ≈ `solo`, so the benefit is about the threat and not about navigation. | `sightings − solo` has a CI including 0 everywhere, **or** the predator-off control shows `sightings` > `solo`. |
| H4 | Imperfect vision costs escapes: `surrogate` < `perfect` for the same condition and cell. | Surrogate ≥ perfect in most cells. |
| H5 | Moving as a group does not beat sharing without grouping (`group` ≤ `share`), because groups are easier to sense and gathering costs time. | `group` > `share` with CI excluding 0 in most cells. |

## Robustness (reported, not predicted)

- Ambush predator policy.
- Communication range limited.
- Predator-off subtraction.
- Group detection treated as one target.

import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { COLORS } from "../constants";
import { INTER, MONO } from "../fonts";
import { Captions, Card, Crop, Headline, SceneBg, Tag, clamp, fadeWindow, useEnter } from "../components";

// RESULTS.md, Tier A v1 (calibrated), grid-mean escape rate
const fog = [
  { k: "clear air", solo: 0.268, share: 0.424 },
  { k: "light fog · 78 m", solo: 0.27, share: 0.419 },
  { k: "thick fog · 39 m", solo: 0.251, share: 0.355 },
];

export const Science: React.FC = () => {
  const frame = useCurrentFrame();
  const pA = fadeWindow(frame, 0, 14, 160, 178);
  const pB = fadeWindow(frame, 170, 188, 410, 428);
  const pC = interpolate(frame, [420, 438], [0, 1], clamp);
  const count = Math.round(interpolate(frame, [30, 120], [0, 159800], { ...clamp, easing: (t) => 1 - Math.pow(1 - t, 3) }));
  const a1 = useEnter(8);
  const a2 = useEnter(24);
  const badge = useEnter(300);
  const heat = useEnter(200);
  return (
    <SceneBg>
      {/* A: pre-registration + scale */}
      <AbsoluteFill style={{ opacity: pA, padding: "80px 110px" }}>
        <Headline size={60}>Pre-registered, then tested</Headline>
        <div style={{ display: "flex", gap: 40, marginTop: 56 }}>
          <div style={a1.style}>
            <Card style={{ width: 780, height: 420 }}>
              <div style={{ fontFamily: MONO, fontSize: 22, color: COLORS.accent, letterSpacing: 2 }}>1 · PREDICTIONS FIRST</div>
              <div style={{ fontFamily: INTER, fontSize: 44, fontWeight: 800, color: COLORS.white, marginTop: 18 }}>PREREGISTRATION.md</div>
              <div style={{ fontFamily: MONO, fontSize: 24, color: COLORS.info, marginTop: 16 }}>SHA-256 790902a3…</div>
              <div style={{ fontFamily: INTER, fontSize: 28, color: COLORS.offWhite, marginTop: 28, lineHeight: 1.45 }}>
                Five predictions, committed before the calibrated runs. Every verdict reported, right or wrong.
              </div>
            </Card>
          </div>
          <div style={a2.style}>
            <Card style={{ width: 780, height: 420 }}>
              <div style={{ fontFamily: MONO, fontSize: 22, color: COLORS.accent, letterSpacing: 2 }}>2 · THEN THE SWEEPS</div>
              <div style={{ fontFamily: INTER, fontSize: 104, fontWeight: 900, color: COLORS.white, marginTop: 6, fontVariantNumeric: "tabular-nums" }}>
                {count.toLocaleString("en-US")}
              </div>
              <div style={{ fontFamily: INTER, fontSize: 30, color: COLORS.offWhite }}>calibrated episodes, paired by seed</div>
              <div style={{ fontFamily: INTER, fontSize: 24, color: COLORS.muted, marginTop: 22, lineHeight: 1.45 }}>
                One AWS Graviton c8g.2xlarge. The 134,400-episode grid took 1,399 s, about $0.08.
              </div>
            </Card>
          </div>
        </div>
      </AbsoluteFill>

      {/* B: headline result */}
      <AbsoluteFill style={{ opacity: pB, padding: "64px 110px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Headline size={54}>Share of hikers who got out</Headline>
          <Tag>Tier A v1 · calibrated perception · simulation</Tag>
        </div>
        <div style={{ display: "flex", gap: 60, marginTop: 40 }}>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 56, height: 520 }}>
            {fog.map((f, i) => {
              const g = interpolate(frame, [186 + i * 14, 230 + i * 14], [0, 1], { ...clamp, easing: (t) => 1 - Math.pow(1 - t, 3) });
              const H = 820;
              return (
                <div key={f.k} style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                  <div style={{ display: "flex", alignItems: "flex-end", gap: 12 }}>
                    {[
                      { v: f.solo, c: COLORS.solo, l: "solo" },
                      { v: f.share, c: COLORS.accent, l: "share" },
                    ].map((b) => (
                      <div key={b.l} style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 112 }}>
                        <div style={{ fontFamily: INTER, fontWeight: 800, fontSize: 34, color: b.c, opacity: g, marginBottom: 8 }}>{(b.v * 100).toFixed(1)}%</div>
                        <div style={{ width: 100, height: H * b.v * g, background: b.c, borderRadius: "8px 8px 0 0" }} />
                        <div style={{ fontFamily: MONO, fontSize: 18, color: COLORS.muted, marginTop: 8 }}>{b.l}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ fontFamily: INTER, fontSize: 24, fontWeight: 600, color: COLORS.offWhite, marginTop: 6 }}>{f.k}</div>
                </div>
              );
            })}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 20, marginLeft: 40 }}>
            <div style={heat.style}>
              <div style={{ background: "#ffffff", borderRadius: 14, padding: 10 }}>
                <Crop src="headline.png" iw={3312} ih={792} x={0} y={0} cw={770} ch={792} width={340} style={{ borderRadius: 6 }} />
              </div>
              <div style={{ fontFamily: MONO, fontSize: 16, color: COLORS.muted, marginTop: 8 }}>Δ vs solo per setting · light fog · docs/img/headline.png</div>
            </div>
            <div style={badge.style}>
              <Card accent={COLORS.accent} style={{ width: 360, padding: 20 }}>
                <div style={{ fontFamily: INTER, fontSize: 64, fontWeight: 900, color: COLORS.accent, lineHeight: 1 }}>56 / 56</div>
                <div style={{ fontFamily: INTER, fontSize: 21, color: COLORS.offWhite, marginTop: 10, lineHeight: 1.35 }}>
                  settings where sharing beat solo (95% CI above 0), at every fog level
                </div>
              </Card>
            </div>
          </div>
        </div>
      </AbsoluteFill>

      {/* C: honesty */}
      <AbsoluteFill style={{ opacity: pC, padding: "80px 110px" }}>
        <Headline size={60}>Limits and a failed prediction</Headline>
        <div style={{ display: "flex", gap: 30, marginTop: 50 }}>
          {[
            { k: "H3 · PARTLY FALSIFIED", c: COLORS.solo, t: "Sharing only predator sightings kept helping even when the predator senses the whole forest. We predicted it would not.", d: 432 },
            { k: "FOG", c: COLORS.info, t: "Thick fog (39 m) cuts the benefit by a third: +15.6 → +10.4 points. Still above zero in all 56 settings.", d: 448 },
            { k: "VISION IS THE BOTTLENECK", c: COLORS.info, t: "Sharing recovers 56% of the gap to perfect perception. Our earlier guessed model said 81%.", d: 464 },
          ].map((x) => {
            const e = useEnter(x.d);
            return (
              <div key={x.k} style={e.style}>
                <Card accent={x.c} style={{ width: 535, height: 400 }}>
                  <div style={{ fontFamily: MONO, fontSize: 21, color: x.c, letterSpacing: 1.6 }}>{x.k}</div>
                  <div style={{ fontFamily: INTER, fontSize: 31, fontWeight: 600, color: COLORS.white, marginTop: 22, lineHeight: 1.38 }}>{x.t}</div>
                </Card>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>

      <Captions
        lines={[
          { from: 6, to: 170, text: "Predictions were committed first. Then 159,800 calibrated episodes ran on AWS Graviton." },
          { from: 176, to: 300, text: "In clear air, sharing raised the escape rate from 27% to 42%." },
          { from: 300, to: 420, text: "Sharing beat solo in all 56 settings, at every fog level tested." },
          { from: 426, to: 598, text: "One prediction was partly wrong: shared sightings kept helping against a predator that senses the whole forest." },
        ]}
      />
    </SceneBg>
  );
};

import React from "react";
import { AbsoluteFill, OffthreadVideo, Sequence, interpolate, staticFile, useCurrentFrame } from "remotion";
import { HERO_RATE } from "../constants";
import { COLORS } from "../constants";
import { INTER, MONO } from "../fonts";
import { Captions, Card, Headline, SceneBg, Tag, clamp, fadeWindow, useEnter } from "../components";

const VW = 1459;
const VH = 821;

export const Hero: React.FC = () => {
  const frame = useCurrentFrame();
  const title = fadeWindow(frame, 0, 14, 62, 80);
  const vid = interpolate(frame, [62, 86], [0, 1], clamp);
  const settle = interpolate(frame, [62, 130], [1.03, 1], clamp);
  // ken-burns: zoom into the two maps mid-episode, back out to show the counters
  const z = interpolate(frame, [230, 290, 520, 575], [1, 1.6, 1.6, 1], { ...clamp, easing: (t) => t * t * (3 - 2 * t) });
  const dim = interpolate(frame, [612, 640], [0, 0.88], clamp);
  const res = interpolate(frame, [622, 642], [0, 1], clamp);
  const t1 = useEnter(4);
  const l = useEnter(626);
  const r = useEnter(640);
  const foot = interpolate(frame, [670, 690], [0, 1], clamp);
  return (
    <SceneBg>
      {/* title */}
      <AbsoluteFill style={{ opacity: title, alignItems: "center", justifyContent: "center", paddingBottom: 120 }}>
        <div style={{ ...t1.style, textAlign: "center" }}>
          <Headline size={76}>Same forest, run twice</Headline>
          <div style={{ display: "flex", gap: 24, justifyContent: "center", marginTop: 36 }}>
            <Tag color={COLORS.solo}>Left · SOLO · private maps</Tag>
            <Tag color={COLORS.accent}>Right · SHARED MAP · one living map</Tag>
          </div>
        </div>
      </AbsoluteFill>

      {/* real viewer capture */}
      <AbsoluteFill style={{ opacity: vid }}>
        <div style={{ position: "absolute", top: 20, left: (1920 - VW) / 2, width: VW, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Tag>REAL VIEWER CAPTURE · seed 45 · calibrated perception · light fog</Tag>
          <div style={{ display: "flex", gap: 14 }}>
            <Tag color={COLORS.solo}>SOLO</Tag>
            <Tag color={COLORS.accent}>SHARED MAP</Tag>
          </div>
        </div>
        <div
          style={{
            position: "absolute", top: 76, left: (1920 - VW) / 2, width: VW, height: VH, borderRadius: 14, overflow: "hidden",
            border: `2px solid ${COLORS.border}`, boxShadow: "0 24px 70px rgba(0,0,0,0.55)", transform: `scale(${settle})`,
          }}
        >
          <div style={{ width: VW, height: VH, transform: `scale(${z})`, transformOrigin: "30% 85%" }}>
            <Sequence from={62} layout="none">
              <OffthreadVideo src={staticFile("hero.mp4")} muted playbackRate={HERO_RATE} style={{ width: VW, height: VH }} />
            </Sequence>
          </div>
          <AbsoluteFill style={{ background: `rgba(4,8,6,${dim})` }} />
        </div>
      </AbsoluteFill>

      {/* outcome */}
      <AbsoluteFill style={{ opacity: res, alignItems: "center", paddingTop: 230 }}>
        <div style={{ display: "flex", gap: 60 }}>
          <div style={l.style}>
            <Card accent={COLORS.solo} style={{ width: 560, textAlign: "center", background: "rgba(15,22,18,0.94)" }}>
              <div style={{ fontFamily: MONO, fontSize: 24, color: COLORS.solo, letterSpacing: 2 }}>SOLO</div>
              <div style={{ fontFamily: INTER, fontSize: 112, fontWeight: 900, color: COLORS.white, lineHeight: 1.1 }}>2 of 4</div>
              <div style={{ fontFamily: INTER, fontSize: 32, color: COLORS.offWhite }}>got out · 2 intercepted</div>
            </Card>
          </div>
          <div style={r.style}>
            <Card accent={COLORS.accent} style={{ width: 560, textAlign: "center", background: "rgba(15,22,18,0.94)" }}>
              <div style={{ fontFamily: MONO, fontSize: 24, color: COLORS.accent, letterSpacing: 2 }}>SHARED MAP</div>
              <div style={{ fontFamily: INTER, fontSize: 112, fontWeight: 900, color: COLORS.accent, lineHeight: 1.1 }}>4 of 4</div>
              <div style={{ fontFamily: INTER, fontSize: 32, color: COLORS.offWhite }}>got out</div>
            </Card>
          </div>
        </div>
        <div style={{ opacity: foot, marginTop: 34, fontFamily: MONO, fontSize: 22, color: COLORS.offWhite, textAlign: "center", lineHeight: 1.5 }}>
          Seed 45 was picked as a clear contrast. It is not the average:<br />
          across the grid at this fog level, sharing adds about +15 points.
        </div>
      </AbsoluteFill>

      <Captions
        lines={[
          { from: 4, to: 84, text: "Same forest, same predator, same random seed. Run twice." },
          { from: 90, to: 250, text: "Left: each hiker keeps a private map. Right: one shared map." },
          { from: 250, to: 420, text: "Sightings, exits and explored ground reach every hiker. Old evidence fades." },
          { from: 420, to: 610, text: "Only a separate evaluator holds ground truth. The hikers never see it." },
          { from: 620, to: 748, text: "Solo: 2 of 4 got out. Shared map: 4 of 4. A selected example, not the average." },
        ]}
      />
    </SceneBg>
  );
};

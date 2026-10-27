import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { COLORS } from "../constants";
import { INTER, MONO } from "../fonts";
import { Captions, HikerDot, SceneBg, Sep, clamp, fadeWindow, useEnter } from "../components";

const DOTS = ["#4ade80", "#7cc4ff", "#c4a7ff", "#f5b041"];

export const Close: React.FC = () => {
  const frame = useCurrentFrame();
  const p1 = fadeWindow(frame, 0, 14, 150, 168);
  const p2 = interpolate(frame, [160, 182], [0, 1], clamp);
  const a = useEnter(8);
  const b = useEnter(34);
  const logo = useEnter(170);
  const tag = useEnter(196);
  const chips = useEnter(214);
  const merge = interpolate(frame, [170, 215], [1, 0], { ...clamp, easing: (t) => 1 - Math.pow(1 - t, 3) });
  const corner = (extra: React.CSSProperties): React.CSSProperties => ({ position: "absolute", width: 56, height: 56, borderColor: COLORS.accent, opacity: p2, ...extra });
  return (
    <SceneBg>
      <AbsoluteFill style={{ opacity: p1, alignItems: "center", justifyContent: "center", paddingBottom: 120 }}>
        <div style={{ display: "flex", gap: 110, alignItems: "flex-end" }}>
          <div style={{ ...a.style, textAlign: "center" }}>
            <div style={{ fontFamily: MONO, fontSize: 28, color: COLORS.solo, letterSpacing: 3 }}>SOLO</div>
            <div style={{ fontFamily: INTER, fontSize: 170, fontWeight: 900, color: COLORS.offWhite, lineHeight: 1 }}>26.8%</div>
          </div>
          <div style={{ ...b.style, fontFamily: INTER, fontSize: 110, color: COLORS.muted, paddingBottom: 20 }}>→</div>
          <div style={{ ...b.style, textAlign: "center" }}>
            <div style={{ fontFamily: MONO, fontSize: 28, color: COLORS.accent, letterSpacing: 3 }}>SHARED MAP</div>
            <div style={{ fontFamily: INTER, fontSize: 170, fontWeight: 900, color: COLORS.accent, lineHeight: 1 }}>42.4%</div>
          </div>
        </div>
        <div style={{ ...b.style, marginTop: 30, fontFamily: INTER, fontSize: 28, color: COLORS.muted }}>
          hikers who got out · clear air · mean over 56 settings × 200 paired forests · simulation
        </div>
      </AbsoluteFill>

      <AbsoluteFill style={{ opacity: p2, alignItems: "center", justifyContent: "center", paddingBottom: 110 }}>
        <div style={corner({ top: 50, left: 50, borderTop: "3px solid", borderLeft: "3px solid" })} />
        <div style={corner({ top: 50, right: 50, borderTop: "3px solid", borderRight: "3px solid" })} />
        <div style={corner({ bottom: 50, left: 50, borderBottom: "3px solid", borderLeft: "3px solid" })} />
        <div style={corner({ bottom: 50, right: 50, borderBottom: "3px solid", borderRight: "3px solid" })} />
        <div style={{ ...logo.style, display: "flex", gap: 14, marginBottom: 26 }}>
          {DOTS.map((c, i) => (
            <div key={c} style={{ transform: `translateX(${(1.5 - i) * 60 * merge * -1}px)` }}>
              <HikerDot n={i + 1} color={c} size={52} />
            </div>
          ))}
        </div>
        <div style={{ ...logo.style, fontFamily: INTER, fontSize: 124, fontWeight: 900, color: COLORS.white, letterSpacing: -3 }}>Lost Together</div>
        <div style={{ ...tag.style, fontFamily: INTER, fontSize: 40, color: COLORS.accent, fontWeight: 600, marginTop: 4 }}>Share what you see.</div>
        <div style={{ ...tag.style, marginTop: 26 }}>
          <Sep width={520} />
        </div>
        <div style={{ ...chips.style, marginTop: 26, fontFamily: MONO, fontSize: 24, color: COLORS.offWhite, letterSpacing: 1 }}>
          OpenCV 5 · AWS Graviton + GPU · pre-registered · all simulated
        </div>
        <div style={{ ...chips.style, marginTop: 14, fontFamily: INTER, fontSize: 26, color: COLORS.white }}>
          Replay viewer, code and results: links in the submission.
        </div>
        <div style={{ ...chips.style, marginTop: 30, fontFamily: INTER, fontSize: 17, color: COLORS.muted, textAlign: "center", lineHeight: 1.5 }}>
          Forest frames: TartanGround (CC BY 4.0) · YOLOX-s: OpenCV Zoo (Apache-2.0) · Animal and hiker photos: public domain, CC0, CC BY 2.0
          <br />A research simulation, not a safety tool. No real people were filmed.
        </div>
      </AbsoluteFill>

      <Captions
        lines={[
          { from: 6, to: 160, text: "Solo: 27% got out. Shared map: 42%." },
          { from: 170, to: 328, text: "Lost Together. All simulated. Every prediction reported." },
        ]}
      />
    </SceneBg>
  );
};

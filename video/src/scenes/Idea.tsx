import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from "remotion";
import { COLORS } from "../constants";
import { INTER, MONO } from "../fonts";
import { Captions, Card, Crop, Headline, HikerDot, SceneBg, Tag, clamp, fadeWindow, useEnter } from "../components";

const HIKER_COLORS = ["#4ade80", "#7cc4ff", "#c4a7ff", "#f5b041"];
const COLS = 20;
const ROWS = 13;
const CELL = 36;
const seeds = [
  { c: 4, r: 4 },
  { c: 14, r: 3 },
  { c: 5, r: 10 },
  { c: 15, r: 10 },
];

const LivingMap: React.FC<{ frame: number }> = ({ frame }) => {
  const cells: React.ReactNode[] = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      let best = 0;
      let bd = 1e9;
      seeds.forEach((s, i) => {
        const d = Math.hypot(s.c - c, s.r - r);
        if (d < bd) { bd = d; best = i; }
      });
      const appear = 30 + bd * 9 + ((c * 7 + r * 13) % 5) * 3;
      const op = interpolate(frame, [appear, appear + 12], [0, 1], clamp);
      cells.push(
        <div key={`${r}-${c}`} style={{ position: "absolute", left: c * CELL, top: r * CELL, width: CELL - 3, height: CELL - 3, borderRadius: 4, background: HIKER_COLORS[best], opacity: op * 0.32 }} />,
      );
    }
  }
  const pings = [
    { c: 11, r: 6, t: 60 },
    { c: 8, r: 2, t: 110 },
  ];
  const exitOp = interpolate(frame, [130, 145], [0, 1], clamp);
  return (
    <div style={{ position: "relative", width: COLS * CELL, height: ROWS * CELL }}>
      {cells}
      {seeds.map((s, i) => (
        <div key={i} style={{ position: "absolute", left: s.c * CELL - 6, top: s.r * CELL - 6 }}>
          <HikerDot n={i + 1} color={HIKER_COLORS[i]} size={42} />
        </div>
      ))}
      {pings.map((p, i) => {
        const age = frame - p.t;
        const op = age < 0 ? 0 : interpolate(age, [0, 8, 90, 160], [0, 1, 0.55, 0.15], clamp);
        const ring = (age % 40) / 40;
        return (
          <div key={i} style={{ position: "absolute", left: p.c * CELL, top: p.r * CELL, opacity: op }}>
            <div style={{ position: "absolute", left: -10, top: -10, width: 56, height: 56, borderRadius: 56, background: COLORS.bear, opacity: 0.85 }} />
            <div style={{ position: "absolute", left: -10 - ring * 30, top: -10 - ring * 30, width: 56 + ring * 60, height: 56 + ring * 60, borderRadius: 200, border: `3px solid ${COLORS.bear}`, opacity: 1 - ring }} />
          </div>
        );
      })}
      <div style={{ position: "absolute", left: (COLS - 2) * CELL, top: (ROWS - 1) * CELL - 4, opacity: exitOp, fontFamily: MONO, fontWeight: 700, fontSize: 22, color: "#08100b", background: COLORS.accent, padding: "4px 10px", borderRadius: 6 }}>EXIT</div>
    </div>
  );
};

export const Idea: React.FC = () => {
  const frame = useCurrentFrame();
  const pA = fadeWindow(frame, 0, 15, 196, 214);
  const pB = interpolate(frame, [206, 224], [0, 1], clamp);
  const h = useEnter(4);
  return (
    <SceneBg>
      <AbsoluteFill style={{ opacity: pA, padding: "70px 90px" }}>
        <div style={h.style}>
          <Headline size={60}>What if they share what they see?</Headline>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 50, marginTop: 44 }}>
          <div style={{ display: "grid", gridTemplateColumns: "250px 250px", gap: 18 }}>
            {[0, 1, 2, 3].map((i) => {
              const e = useEnter(14 + i * 7);
              return (
                <div key={i} style={{ ...e.style, position: "relative" }}>
                  <Img src={staticFile(`view0_${i}.jpg`)} style={{ width: 250, height: 250, borderRadius: 12, border: `3px solid ${HIKER_COLORS[i]}` }} />
                  <div style={{ position: "absolute", left: 10, top: 10 }}>
                    <HikerDot n={i + 1} color={HIKER_COLORS[i]} size={36} />
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ fontFamily: INTER, fontSize: 80, color: COLORS.accent, opacity: interpolate(frame, [40, 60], [0, 1], clamp) }}>→</div>
          <div>
            <div style={{ fontFamily: MONO, fontSize: 22, color: COLORS.accent, letterSpacing: 2, marginBottom: 14 }}>ONE SHARED LIVING MAP</div>
            <Card style={{ padding: 18 }}>
              <LivingMap frame={frame} />
            </Card>
            <div style={{ fontFamily: MONO, fontSize: 19, color: COLORS.muted, marginTop: 12 }}>
              <span style={{ color: COLORS.bear }}>●</span> sightings fade with age · terrain and exits stay
            </div>
          </div>
        </div>
      </AbsoluteFill>

      <AbsoluteFill style={{ opacity: pB, padding: "70px 90px" }}>
        <Headline size={60}>A teammate's report triggers a closer look</Headline>
        <div style={{ marginTop: 18 }}>
          <Tag>REAL YOLOX OUTPUT · bear at 20 m · one example, not a rate</Tag>
        </div>
        <div style={{ display: "flex", gap: 40, marginTop: 30, alignItems: "flex-start" }}>
          <Card style={{ width: 470 }}>
            <div style={{ fontFamily: MONO, fontSize: 20, color: COLORS.muted, marginBottom: 14 }}>SHARED MAP · NEW REPORT</div>
            <div style={{ fontFamily: INTER, fontSize: 32, fontWeight: 700, color: COLORS.white, lineHeight: 1.3 }}>
              Hiker 2 <span style={{ color: COLORS.muted, fontWeight: 500 }}>reported something nearby</span>
            </div>
            <div style={{ height: 24 }} />
            <div style={{ fontFamily: INTER, fontSize: 32, fontWeight: 700, color: COLORS.accent, lineHeight: 1.3, opacity: interpolate(frame, [240, 255], [0, 1], clamp) }}>
              Hiker 4 scans that way with a 3× zoom
            </div>
            <div style={{ marginTop: 26, fontFamily: MONO, fontSize: 19, color: COLORS.muted, lineHeight: 1.5, opacity: interpolate(frame, [255, 270], [0, 1], clamp) }}>
              crop → zoom → re-detect<br />on the same OpenCV 5 pipeline
            </div>
          </Card>
          {[0, 1].map((col) => {
            const e = useEnter(236 + col * 40);
            return (
              <div key={col} style={{ ...e.style, display: "flex", flexDirection: "column", alignItems: "center" }}>
                <Crop src="fog_strip.jpg" iw={1260} ih={1260} x={col * 420} y={858} cw={420} ch={402} width={560} style={{ border: `2px solid ${col === 0 ? COLORS.accent : COLORS.border}` }} />
                <div style={{ marginTop: 14, fontFamily: INTER, fontWeight: 800, fontSize: 32, color: col === 0 ? COLORS.accent : COLORS.muted }}>
                  {col === 0 ? "3× tile, clear air → bear 0.81" : "3× tile, light fog → missed"}
                </div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>

      <Captions
        lines={[
          { from: 6, to: 92, text: "Does sharing what each hiker sees help them get out?" },
          { from: 92, to: 198, text: "Each camera view feeds one shared, living map. Old sightings fade." },
          { from: 202, to: 300, text: "A teammate's report tells a hiker where to look closer, with a 3× zoom." },
          { from: 300, to: 388, text: "Clear air: bear found at 20 metres. Light fog: missed." },
        ]}
      />
    </SceneBg>
  );
};

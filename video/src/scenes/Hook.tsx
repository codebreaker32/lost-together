import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from "remotion";
import { COLORS } from "../constants";
import { INTER, MONO } from "../fonts";
import { Captions, Crop, Headline, HikerDot, SceneBg, Tag, Vignette, clamp, fadeWindow, useEnter } from "../components";

const TILE = 420; // fog_strip.jpg is a 3x3 grid of 420 px tiles
const tiles = [
  { col: 0, note: "clear air", det: "bear 0.82", color: COLORS.accent },
  { col: 1, note: "light fog · 78 m visibility", det: "bear 0.71", color: COLORS.accent },
  { col: 2, note: "thick fog · 39 m visibility", det: "“elephant” 0.35", color: COLORS.solo },
];

const rates = [
  { d: "5 m", p: 0.97 },
  { d: "10 m", p: 0.8 },
  { d: "15 m", p: 0.04 },
  { d: "20 m", p: 0 },
];

export const Hook: React.FC = () => {
  const frame = useCurrentFrame();
  const pA = fadeWindow(frame, 0, 18, 140, 160);
  const pB = fadeWindow(frame, 150, 170, 320, 340);
  const pC = interpolate(frame, [332, 352], [0, 1], clamp);
  const row30 = interpolate(frame, [246, 262], [0, 1], clamp);
  const zoom = interpolate(frame, [0, 160], [1.12, 1.0], clamp);
  const hikers = [
    { n: 1, x: 640, y: 640, d: 22 },
    { n: 2, x: 820, y: 700, d: 30 },
    { n: 3, x: 1050, y: 660, d: 38 },
    { n: 4, x: 1230, y: 720, d: 46 },
  ];
  return (
    <SceneBg>
      {/* Phase A: the forest */}
      <AbsoluteFill style={{ opacity: pA }}>
        <Img src={staticFile("forest_bg.jpg")} style={{ width: 1920, height: 1920, objectFit: "cover", position: "absolute", top: -420, transform: `scale(${zoom})`, filter: "brightness(0.55) saturate(0.9)" }} />
        <Vignette strength={0.85} />
        <AbsoluteFill style={{ padding: 90 }}>
          <Tag color={COLORS.offWhite}>TartanGround forest frame · simulated</Tag>
        </AbsoluteFill>
        {hikers.map((h) => {
          const e = useEnter(h.d);
          return (
            <div key={h.n} style={{ position: "absolute", left: h.x, top: h.y, ...e.style }}>
              <HikerDot n={h.n} color={COLORS.solo} size={58} />
            </div>
          );
        })}
        <div style={{ position: "absolute", left: 1460, top: 470, opacity: interpolate(frame, [80, 100], [0, 1], clamp) * (0.6 + 0.4 * Math.sin(frame / 7)) }}>
          <div style={{ width: 70, height: 70, borderRadius: 70, border: `3px dashed ${COLORS.bear}`, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: INTER, fontWeight: 800, fontSize: 38, color: COLORS.bear }}>?</div>
        </div>
      </AbsoluteFill>

      {/* Phase B: real YOLOX output, fog strip */}
      <AbsoluteFill style={{ opacity: pB, alignItems: "center", paddingTop: 70 }}>
        <Tag>REAL YOLOX OUTPUT · OpenCV 5 · cv2.dnn ENGINE_NEW</Tag>
        <div style={{ display: "flex", gap: 34, marginTop: 34 }}>
          {tiles.map((t, i) => {
            const e = useEnter(160 + i * 18);
            return (
              <div key={t.col} style={{ ...e.style, display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div style={{ position: "relative", width: 540, height: 540 }}>
                  <Crop src="fog_strip.jpg" iw={1260} ih={1260} x={t.col * TILE} y={0} cw={TILE} ch={TILE} width={540} style={{ position: "absolute", border: `2px solid ${COLORS.border}` }} />
                  <Crop src="fog_strip.jpg" iw={1260} ih={1260} x={t.col * TILE} y={TILE} cw={TILE} ch={TILE} width={540} style={{ position: "absolute", opacity: row30, border: `2px solid ${COLORS.border}` }} />
                </div>
                <div style={{ marginTop: 16, fontFamily: MONO, fontSize: 22, color: COLORS.muted }}>{t.note}</div>
                <div style={{ position: "relative", height: 48, width: 540 }}>
                  <div style={{ position: "absolute", width: "100%", textAlign: "center", opacity: 1 - row30, fontFamily: INTER, fontWeight: 800, fontSize: 36, color: t.color }}>
                    bear at 10 m → {t.det}
                  </div>
                  <div style={{ position: "absolute", width: "100%", textAlign: "center", opacity: row30, fontFamily: INTER, fontWeight: 800, fontSize: 36, color: COLORS.muted }}>
                    bear at 30 m → nothing
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>

      {/* Phase C: calibrated detection range */}
      <AbsoluteFill style={{ opacity: pC, alignItems: "center", paddingTop: 110 }}>
        <Headline size={58} style={{ textAlign: "center" }}>
          Detection range of a 640 px camera
        </Headline>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 70, height: 470, marginTop: 50 }}>
          {rates.map((r, i) => {
            const grow = interpolate(frame, [350 + i * 10, 395 + i * 10], [0, 1], clamp);
            const h = Math.max(6, 400 * r.p * grow);
            return (
              <div key={r.d} style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 190 }}>
                <div style={{ fontFamily: INTER, fontWeight: 800, fontSize: 46, color: r.p > 0.5 ? COLORS.accent : COLORS.bear, marginBottom: 12, opacity: grow }}>
                  {Math.round(r.p * 100)}%
                </div>
                <div style={{ width: 150, height: h, borderRadius: "10px 10px 0 0", background: r.p > 0.5 ? COLORS.accent : COLORS.bear, opacity: 0.9 }} />
                <div style={{ marginTop: 14, fontFamily: MONO, fontSize: 30, color: COLORS.offWhite }}>{r.d}</div>
              </div>
            );
          })}
        </div>
        <div style={{ marginTop: 20, fontFamily: MONO, fontSize: 19, color: COLORS.muted }}>
          detection rate · YOLOX-s, score ≥ 0.25, clear air · results/calibration.json
        </div>
      </AbsoluteFill>

      <Captions
        lines={[
          { from: 6, to: 72, text: "Four hikers are lost in a forest." },
          { from: 72, to: 152, text: "A predator is nearby. Each hiker has one camera." },
          { from: 156, to: 248, text: "At 10 metres, the detector finds the bear. In thick fog, it reports an elephant." },
          { from: 248, to: 330, text: "At 30 metres, it finds nothing." },
          { from: 336, to: 476, text: "Calibrated detection range: about 15 metres." },
        ]}
      />
    </SceneBg>
  );
};

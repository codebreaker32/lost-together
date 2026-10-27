import React from "react";
import { AbsoluteFill, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { COLORS } from "./constants";
import { INTER, MONO } from "./fonts";

export const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

/** Spring entrance: opacity + small rise + subtle scale (never from 0). */
export function useEnter(delay: number, opts: { rise?: number; damping?: number; stiffness?: number } = {}) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = spring({ frame: frame - delay, fps, config: { damping: opts.damping ?? 18, stiffness: opts.stiffness ?? 150 } });
  return {
    p,
    style: {
      opacity: interpolate(p, [0, 0.35], [0, 1], clamp),
      transform: `translateY(${interpolate(p, [0, 1], [opts.rise ?? 24, 0])}px) scale(${interpolate(p, [0, 1], [0.95, 1])})`,
    } as React.CSSProperties,
  };
}

/** Fade between [in0,in1] and out at [out0,out1]. */
export function fadeWindow(frame: number, in0: number, in1: number, out0: number, out1: number) {
  return interpolate(frame, [in0, in1, out0, out1], [0, 1, 1, 0], clamp);
}

export const SceneBg: React.FC<{ tint?: string; children?: React.ReactNode }> = ({ tint = "#0c1d14", children }) => (
  <AbsoluteFill style={{ background: `radial-gradient(ellipse at 50% 35%, ${tint} 0%, ${COLORS.bg} 68%)` }}>
    {children}
  </AbsoluteFill>
);

export const Vignette: React.FC<{ strength?: number }> = ({ strength = 0.75 }) => (
  <AbsoluteFill style={{ background: `radial-gradient(ellipse at 50% 50%, rgba(0,0,0,0) 45%, rgba(0,0,0,${strength}) 100%)`, pointerEvents: "none" }} />
);

export type Cap = { from: number; to: number; text: string };

/** Narration captions: one line block at the bottom, carries the story. */
export const Captions: React.FC<{ lines: Cap[] }> = ({ lines }) => {
  const frame = useCurrentFrame();
  const cur = lines.find((l) => frame >= l.from && frame < l.to);
  if (!cur) return null;
  const op = interpolate(frame, [cur.from, cur.from + 8, cur.to - 8, cur.to], [0, 1, 1, 0], clamp);
  const y = interpolate(frame, [cur.from, cur.from + 10], [10, 0], clamp);
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 54, pointerEvents: "none", zIndex: 50 }}>
      <div
        style={{
          opacity: op,
          transform: `translateY(${y}px)`,
          maxWidth: 1560,
          background: "rgba(4,8,6,0.82)",
          border: `1px solid ${COLORS.border}`,
          borderRadius: 16,
          padding: "16px 34px",
          fontFamily: INTER,
          fontSize: 40,
          lineHeight: 1.28,
          fontWeight: 600,
          color: COLORS.white,
          textAlign: "center",
          letterSpacing: -0.2,
          boxShadow: "0 10px 40px rgba(0,0,0,0.45)",
        }}
      >
        {cur.text}
      </div>
    </AbsoluteFill>
  );
};

/** Small mono label used to mark real footage / real output. */
export const Tag: React.FC<{ children: React.ReactNode; color?: string; style?: React.CSSProperties }> = ({ children, color = COLORS.accent, style }) => (
  <div
    style={{
      display: "inline-flex",
      alignSelf: "flex-start",
      alignItems: "center",
      gap: 10,
      fontFamily: MONO,
      fontSize: 17,
      fontWeight: 700,
      letterSpacing: 1.6,
      textTransform: "uppercase",
      color,
      background: "rgba(0,0,0,0.6)",
      border: `1.5px solid ${color}`,
      borderRadius: 8,
      padding: "7px 14px",
      ...style,
    }}
  >
    <span style={{ width: 9, height: 9, borderRadius: 9, background: color, display: "inline-block" }} />
    {children}
  </div>
);

/** Show a sub-rectangle (x,y,cw,ch) of an image of natural size iw x ih, scaled to `width`. */
export const Crop: React.FC<{
  src: string; iw: number; ih: number; x: number; y: number; cw: number; ch: number; width: number;
  style?: React.CSSProperties; children?: React.ReactNode;
}> = ({ src, iw, ih, x, y, cw, ch, width, style, children }) => {
  const s = width / cw;
  return (
    <div style={{ width, height: ch * s, overflow: "hidden", position: "relative", borderRadius: 12, ...style }}>
      <Img src={staticFile(src)} style={{ position: "absolute", left: -x * s, top: -y * s, width: iw * s, height: ih * s, maxWidth: "none" }} />
      {children}
    </div>
  );
};

export const Card: React.FC<{ children: React.ReactNode; style?: React.CSSProperties; accent?: string }> = ({ children, style, accent }) => (
  <div
    style={{
      background: COLORS.bgCard,
      border: `1.5px solid ${accent ?? COLORS.border}`,
      borderRadius: 18,
      padding: 28,
      boxShadow: "0 18px 50px rgba(0,0,0,0.35)",
      ...style,
    }}
  >
    {children}
  </div>
);

export const Headline: React.FC<{ children: React.ReactNode; size?: number; style?: React.CSSProperties }> = ({ children, size = 54, style }) => (
  <div style={{ fontFamily: INTER, fontSize: size, fontWeight: 800, color: COLORS.white, letterSpacing: -1.2, lineHeight: 1.08, ...style }}>{children}</div>
);

export const Sep: React.FC<{ width?: number; color?: string }> = ({ width = 260, color = COLORS.accent }) => (
  <div style={{ width, height: 2, background: `linear-gradient(90deg, transparent, ${color}, transparent)` }} />
);

/** Detection box drawn over an image (coords in image pixels, scaled by s). */
export const DetBox: React.FC<{ box: number[]; s: number; label: string; color: string; opacity?: number; thick?: number }> = ({ box, s, label, color, opacity = 1, thick = 3 }) => {
  const [x, y, w, h] = box;
  return (
    <div style={{ position: "absolute", left: x * s, top: y * s, width: w * s, height: h * s, border: `${thick}px solid ${color}`, borderRadius: 4, opacity }}>
      <div
        style={{
          position: "absolute", bottom: "100%", left: -thick, marginBottom: 4, whiteSpace: "nowrap",
          background: color, color: "#08100b", fontFamily: MONO, fontWeight: 700, fontSize: 18, padding: "2px 8px", borderRadius: 4,
        }}
      >
        {label}
      </div>
    </div>
  );
};

export const HikerDot: React.FC<{ n: number; color: string; size?: number; style?: React.CSSProperties }> = ({ n, color, size = 44, style }) => (
  <div
    style={{
      width: size, height: size, borderRadius: size, background: color, color: "#08100b",
      display: "flex", alignItems: "center", justifyContent: "center",
      fontFamily: INTER, fontWeight: 800, fontSize: size * 0.45, boxShadow: `0 0 0 4px rgba(0,0,0,0.5), 0 0 24px ${color}66`, ...style,
    }}
  >
    {n}
  </div>
);

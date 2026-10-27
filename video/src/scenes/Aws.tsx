import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { COLORS } from "../constants";
import { INTER, MONO } from "../fonts";
import { Captions, Headline, SceneBg, Tag, clamp, useEnter } from "../components";

const Node: React.FC<{ title: string; sub: string; lines: string[]; delay: number; color: string }> = ({ title, sub, lines, delay, color }) => {
  const e = useEnter(delay);
  return (
    <div style={{ ...e.style, background: COLORS.bgCard2, border: `1.5px solid ${color}`, borderRadius: 14, padding: "18px 22px" }}>
      <div style={{ fontFamily: INTER, fontSize: 30, fontWeight: 800, color: COLORS.white }}>{title}</div>
      <div style={{ fontFamily: MONO, fontSize: 19, color, marginTop: 4 }}>{sub}</div>
      {lines.map((l) => (
        <div key={l} style={{ fontFamily: INTER, fontSize: 22, color: COLORS.offWhite, marginTop: 8 }}>· {l}</div>
      ))}
    </div>
  );
};

const Account: React.FC<{ label: string; children: React.ReactNode; delay: number }> = ({ label, children, delay }) => {
  const e = useEnter(delay);
  return (
    <div style={{ ...e.style, width: 780, border: `2px dashed ${COLORS.border}`, borderRadius: 22, padding: 26, position: "relative" }}>
      <div style={{ fontFamily: MONO, fontSize: 20, color: COLORS.muted, letterSpacing: 1.5, marginBottom: 18 }}>{label}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>{children}</div>
    </div>
  );
};

export const Aws: React.FC = () => {
  const frame = useCurrentFrame();
  const h = useEnter(2);
  const arrow = interpolate(frame, [70, 90], [0, 1], clamp);
  const cap = useEnter(150);
  return (
    <SceneBg tint="#1a1408">
      <AbsoluteFill style={{ padding: "54px 100px" }}>
        <div style={{ ...h.style, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Headline size={54}>Built and run on AWS</Headline>
          <Tag color={COLORS.solo}>ap-south-1 · two accounts</Tag>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 20, marginTop: 34 }}>
          <Account label="MAIN ACCOUNT" delay={10}>
            <Node delay={22} color={COLORS.solo} title="EC2 c8g.2xlarge · Graviton" sub="simulation sweeps" lines={["about 60 episodes per second", "stock OpenCV 5 wheel (COOL not used)"]} />
            <Node delay={36} color={COLORS.info} title="AWS Lambda" sub="function URL" lines={["hosts the replay viewer"]} />
            <Node delay={50} color={COLORS.muted} title="IAM" sub="least privilege" lines={["scoped dev user, no admin rights"]} />
          </Account>
          <div style={{ opacity: arrow, textAlign: "center", width: 120 }}>
            <div style={{ fontFamily: INTER, fontSize: 60, color: COLORS.accent }}>⇄</div>
            <div style={{ fontFamily: MONO, fontSize: 16, color: COLORS.muted }}>SSH tunnel</div>
          </div>
          <Account label="GPU ACCOUNT" delay={60}>
            <Node delay={72} color={COLORS.accent} title="EC2 g6e.xlarge · NVIDIA L40S" sub="qwen3-vl via Ollama" lines={["model port never exposed", "auto-shutdown timer on every launch"]} />
            <div style={cap.style}>
              <div style={{ background: "rgba(74,222,128,0.08)", border: `2px solid ${COLORS.accent}`, borderRadius: 14, padding: "18px 22px" }}>
                <div style={{ fontFamily: INTER, fontSize: 30, fontWeight: 800, color: COLORS.white }}>AWS Budgets · $10 GPU cap</div>
                <div style={{ display: "flex", alignItems: "center", gap: 16, marginTop: 14 }}>
                  <div style={{ width: 440, height: 22, background: COLORS.border, borderRadius: 11, overflow: "hidden" }}>
                    <div style={{ width: 440 * 0.26 * interpolate(frame, [160, 200], [0, 1], clamp), height: 22, background: COLORS.accent }} />
                  </div>
                  <div style={{ fontFamily: INTER, fontSize: 28, fontWeight: 800, color: COLORS.accent }}>~$2.60 used</div>
                </div>
                <div style={{ fontFamily: MONO, fontSize: 16, color: COLORS.muted, marginTop: 10 }}>estimate from instance minutes · alerts at 50 / 80 / 100%</div>
              </div>
            </div>
          </Account>
        </div>
      </AbsoluteFill>
      <Captions
        lines={[
          { from: 4, to: 150, text: "AWS: Graviton for sweeps, an L40S GPU for the agent, Lambda for the viewer." },
          { from: 150, to: 298, text: "GPU spend is capped at $10; about $2.60 used. GPU instances stop themselves." },
        ]}
      />
    </SceneBg>
  );
};

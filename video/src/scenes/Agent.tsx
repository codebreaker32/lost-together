import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from "remotion";
import { COLORS } from "../constants";
import { INTER, MONO } from "../fonts";
import { Captions, Card, Crop, DetBox, Headline, SceneBg, Tag, clamp, fadeWindow, useEnter } from "../components";

const IMG = 600;
const S = IMG / 640;

// viewer/data/agent/s2035_t20_h1_b0.05_llm.json (live Tier C run)
const steps = [
  { t: 16, k: "observe", v: "3 candidates · #0 unknown 0.43 · #1 unknown 0.41 · #2 bear 0.16", c: COLORS.offWhite },
  { t: 70, k: "query_map()", v: "risk near hiker 1.86 · escape threshold 0.9", c: COLORS.info },
  { t: 120, k: "compare_frames(#2)", v: "moving · 1.0 px after removing camera motion", c: COLORS.info },
  { t: 170, k: "look_closer(#2)", v: "defog + contrast + zoom → bear 0.16 → 0.58", c: COLORS.info },
  { t: 232, k: "decision", v: "ESCAPE · truth (evaluator only): a real bear", c: COLORS.accent },
];

const term = [
  { t: 345, c: COLORS.muted, s: "$ uv run python -m lost_together.agent.loop results/tierC/points_a.jsonl --index 27 --mode replay --autonomy advise --approver cli" },
  { t: 356, c: COLORS.solo, s: "=== RANGER CONSOLE: pending ESCAPE order for s2035_t20_h1_b0.05 ===" },
  { t: 366, c: COLORS.offWhite, s: "  agent: #0 unknown 0.43 -> map risk 1.86 -> compare_frames(#2) -> moving 1.0px -> FALLBACK (model call failed: CacheMiss) -> map risk 1.86 -> look_closer -> unknown 0.66 -> compare_frames -> static 0.0px -> ESCAPE" },
  { t: 378, c: COLORS.offWhite, s: "  why:   evidence 1.98 = map risk 1.86 + unknown 0.66 x w0.60 (static)" },
  { t: 388, c: COLORS.info, s: "  [a]pprove  [r]eject -> regroup  or type an action ['keep_going', 'look_again', 'regroup', 'escape']  (30s, then approve): a" },
  { t: 402, c: COLORS.accent, s: "[llm ] ... -> ranger: approved -> ESCAPE   (source=fallback, 0.25s)" },
];

const rows = [
  { k: "False alarms", sub: "of 192 false sightings", r: "110", a: "57", good: true },
  { k: "Late escapes", sub: "of 168 real bears", r: "42", a: "85", good: false },
  { k: "Thick fog: bears missed", sub: "of 66 real bears at 39 m visibility", r: "27", a: "46", good: false },
];

export const Agent: React.FC = () => {
  const frame = useCurrentFrame();
  const pA = fadeWindow(frame, 0, 14, 322, 340);
  const pB = fadeWindow(frame, 332, 348, 446, 462);
  const pC = interpolate(frame, [454, 470], [0, 1], clamp);
  const img = useEnter(6);
  const focus2 = interpolate(frame, [120, 132], [0, 1], clamp);
  const zoomIn = useEnter(176);
  const verdict = useEnter(560);
  return (
    <SceneBg tint="#14140b">
      {/* A: real trace */}
      <AbsoluteFill style={{ opacity: pA, padding: "52px 90px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Headline size={52}>Uncertain sightings: the agent looks again</Headline>
          <Tag color={COLORS.solo}>REAL AGENT TRACE · qwen3-vl:30b on AWS g6e (L40S)</Tag>
        </div>
        <div style={{ display: "flex", gap: 44, marginTop: 34 }}>
          <div style={{ ...img.style, position: "relative", width: IMG, height: IMG }}>
            <Img src={staticFile("agent_s2035.jpg")} style={{ width: IMG, height: IMG, borderRadius: 12 }} />
            <DetBox box={[238, 382, 71, 48]} s={S} label="#0 unknown 0.43" color={COLORS.muted} opacity={1 - 0.6 * focus2} />
            <DetBox box={[137, 351, 74, 40]} s={S} label="#1 unknown 0.41" color={COLORS.muted} opacity={1 - 0.6 * focus2} />
            <DetBox box={[357, 308, 40, 29]} s={S} label="#2 bear 0.16" color={focus2 > 0.5 ? COLORS.solo : COLORS.muted} thick={focus2 > 0.5 ? 4 : 3} />
            <div style={{ position: "absolute", right: 14, top: 14, ...zoomIn.style }}>
              <Crop src="agent_s2035.jpg" iw={640} ih={640} x={337} y={282} cw={80} ch={80} width={230} style={{ border: `3px solid ${COLORS.accent}` }}>
                <div style={{ position: "absolute", left: 0, bottom: 0, background: COLORS.accent, color: "#08100b", fontFamily: MONO, fontWeight: 700, fontSize: 18, padding: "2px 8px" }}>look_closer · bear 0.58</div>
              </Crop>
            </div>
            <div style={{ position: "absolute", left: 12, bottom: 12, fontFamily: MONO, fontSize: 16, color: COLORS.offWhite, background: "rgba(0,0,0,0.6)", padding: "3px 8px", borderRadius: 6 }}>
              composited TartanGround frame · light fog
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 16, width: 1000 }}>
            {steps.map((s, i) => {
              const e = useEnter(s.t);
              const last = i === steps.length - 1;
              return (
                <div key={s.k} style={e.style}>
                  <div style={{ background: last ? "rgba(74,222,128,0.10)" : COLORS.bgCard, border: `1.5px solid ${last ? COLORS.accent : COLORS.border}`, borderRadius: 14, padding: "16px 22px" }}>
                    <div style={{ fontFamily: MONO, fontSize: 24, fontWeight: 700, color: s.c }}>{s.k}</div>
                    <div style={{ fontFamily: INTER, fontSize: 27, color: last ? COLORS.accent : COLORS.white, marginTop: 4, fontWeight: last ? 800 : 500 }}>{s.v}</div>
                  </div>
                </div>
              );
            })}
            <div style={{ fontFamily: MONO, fontSize: 17, color: COLORS.muted }}>tools take candidate IDs, never pixel coordinates · ≤ 3 tool turns · strict JSON</div>
          </div>
        </div>
      </AbsoluteFill>

      {/* B: ranger console */}
      <AbsoluteFill style={{ opacity: pB, padding: "60px 90px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Headline size={52}>A ranger approves escape orders</Headline>
          <Tag>REAL TERMINAL OUTPUT · ranger console</Tag>
        </div>
        <div style={{ marginTop: 30, background: "#05070a", border: `1.5px solid ${COLORS.border}`, borderRadius: 14, overflow: "hidden" }}>
          <div style={{ display: "flex", gap: 9, padding: "12px 16px", background: "#10151a" }}>
            {["#f87171", "#fbbf24", "#4ade80"].map((c) => <div key={c} style={{ width: 13, height: 13, borderRadius: 13, background: c }} />)}
          </div>
          <div style={{ padding: "18px 24px", fontFamily: MONO, fontSize: 21, lineHeight: 1.5, minHeight: 470 }}>
            {term.map((l) => (
              <div key={l.t} style={{ color: l.c, opacity: interpolate(frame, [l.t, l.t + 6], [0, 1], clamp), whiteSpace: "pre-wrap", marginBottom: 6 }}>{l.s}</div>
            ))}
          </div>
        </div>
        <div style={{ marginTop: 16, fontFamily: INTER, fontSize: 22, color: COLORS.offWhite, opacity: interpolate(frame, [404, 416], [0, 1], clamp) }}>
          Offline replay on a laptop: the cached model answer was missing, so the agent logged a <span style={{ color: COLORS.solo, fontWeight: 700 }}>FALLBACK</span> to the rule. It never acts on a broken answer.
        </div>
      </AbsoluteFill>

      {/* C: honest scoreboard */}
      <AbsoluteFill style={{ opacity: pC, padding: "80px 160px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Headline size={52}>Agent vs rule</Headline>
          <Tag color={COLORS.info}>Tier C · 360 live decisions</Tag>
        </div>
        <div style={{ marginTop: 30, display: "grid", gridTemplateColumns: "1fr 240px 240px", rowGap: 36, alignItems: "center" }}>
          <div />
          <div style={{ fontFamily: MONO, fontSize: 22, color: COLORS.muted, textAlign: "center" }}>RULE</div>
          <div style={{ fontFamily: MONO, fontSize: 22, color: COLORS.muted, textAlign: "center" }}>LLM AGENT</div>
          {rows.map((r, i) => {
            const e = useEnter(470 + i * 14);
            return (
              <React.Fragment key={r.k}>
                <div style={e.style}>
                  <div style={{ fontFamily: INTER, fontSize: 40, fontWeight: 700, color: COLORS.white }}>{r.k}</div>
                  <div style={{ fontFamily: INTER, fontSize: 24, color: COLORS.muted }}>{r.sub}</div>
                </div>
                <div style={{ ...e.style, fontFamily: INTER, fontSize: 72, fontWeight: 900, color: COLORS.offWhite, textAlign: "center" }}>{r.r}</div>
                <div style={{ ...e.style, fontFamily: INTER, fontSize: 72, fontWeight: 900, color: r.good ? COLORS.accent : COLORS.bear, textAlign: "center" }}>{r.a}</div>
              </React.Fragment>
            );
          })}
        </div>
        <div style={{ ...verdict.style, marginTop: 50 }}>
          <Card accent={COLORS.solo} style={{ padding: "18px 26px" }}>
            <div style={{ fontFamily: INTER, fontSize: 30, fontWeight: 700, color: COLORS.white }}>
              Not more accurate overall <span style={{ color: COLORS.muted, fontWeight: 500 }}>(McNemar p = 0.39)</span>. A different trade-off, with 0% fallbacks.
            </div>
          </Card>
        </div>
      </AbsoluteFill>

      <Captions
        lines={[
          { from: 4, to: 110, text: "On uncertain sightings, an LLM agent calls OpenCV tools." },
          { from: 110, to: 232, text: "It queries the map, checks for motion, and looks closer: bear 0.16 becomes bear 0.58." },
          { from: 232, to: 330, text: "It orders an escape. The evaluator confirms a real bear." },
          { from: 334, to: 456, text: "A human ranger approves every escape order. If the model fails, the rule decides." },
          { from: 460, to: 628, text: "The agent is not more accurate than the rule: fewer false alarms, more missed bears in thick fog." },
        ]}
      />
    </SceneBg>
  );
};

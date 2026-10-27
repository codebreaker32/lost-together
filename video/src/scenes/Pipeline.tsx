import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { COLORS } from "../constants";
import { INTER, MONO } from "../fonts";
import { Captions, Card, Crop, Headline, SceneBg, Tag, clamp, useEnter } from "../components";

const Step: React.FC<{ n: number; title: string; api: string; delay: number; children: React.ReactNode; note: React.ReactNode }> = ({ n, title, api, delay, children, note }) => {
  const e = useEnter(delay);
  return (
    <div style={e.style}>
      <Card style={{ width: 846, height: 318, display: "flex", gap: 28, padding: 24 }}>
        <div style={{ flexShrink: 0 }}>{children}</div>
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
          <div style={{ fontFamily: MONO, fontSize: 20, color: COLORS.accent, letterSpacing: 2 }}>STEP {n}</div>
          <div style={{ fontFamily: INTER, fontSize: 40, fontWeight: 800, color: COLORS.white, marginTop: 6 }}>{title}</div>
          <div style={{ fontFamily: MONO, fontSize: 20, color: COLORS.info, marginTop: 12, lineHeight: 1.4 }}>{api}</div>
          <div style={{ fontFamily: INTER, fontSize: 22, color: COLORS.offWhite, marginTop: 14, lineHeight: 1.4 }}>{note}</div>
        </div>
      </Card>
    </div>
  );
};

export const Pipeline: React.FC = () => {
  const frame = useCurrentFrame();
  const h = useEnter(2);
  const bar = (v: number, d: number) => interpolate(frame, [d, d + 30], [0, v], clamp);
  return (
    <SceneBg tint="#0b1a22">
      <AbsoluteFill style={{ padding: "54px 90px" }}>
        <div style={{ ...h.style, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <Headline size={56}>Under the hood: OpenCV 5.0.0</Headline>
          <Tag color={COLORS.info}>no PyTorch · no ONNX Runtime</Tag>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "846px 846px", gap: 26, marginTop: 34 }}>
          <Step n={1} title="Detect" api="YOLOX-s · cv2.dnn ENGINE_NEW · NMSBoxes" delay={14} note="Plus a classical dark-blob pass, so odd shapes become 'unknown' candidates.">
            <Crop src="fog_strip.jpg" iw={1260} ih={1260} x={70} y={140} cw={240} ch={240} width={270} />
          </Step>
          <Step n={2} title="Look closer" api="crop · dark-channel defog · CLAHE · zoom · re-detect" delay={24} note="Triggered by an uncertain sighting or a teammate's report.">
            <Crop src="fog_strip.jpg" iw={1260} ih={1260} x={120} y={980} cw={200} ch={200} width={270} />
          </Step>
          <Step n={3} title="Depth" api="StereoSGBM (3-way), 0.25 m baseline" delay={150} note={<>Median error <b>0.10 m</b> vs ground truth (clear air, pixels closer than 20 m).</>}>
            <div style={{ display: "flex", gap: 8 }}>
              <Crop src="stereo_depth.jpg" iw={1920} ih={640} x={0} y={160} cw={320} ch={480} width={132} />
              <Crop src="stereo_depth.jpg" iw={1920} ih={640} x={1520} y={160} cw={320} ch={480} width={132} />
            </div>
          </Step>
          <Step n={4} title="Where am I?" api="ORB + BFMatcher + solvePnPRansac" delay={290} note={<>Dense <span style={{ fontFamily: MONO }}>ptcloud</span> odometry slid on foliage; PnP gates it.</>}>
            <div style={{ width: 270, height: 270, display: "flex", flexDirection: "column", justifyContent: "center", gap: 22 }}>
              <div style={{ fontFamily: MONO, fontSize: 16, color: COLORS.muted }}>drift, % of path (GT depth)</div>
              {[
                { k: "dense RGB-D", v: 88.5, c: COLORS.bear, d: 300 },
                { k: "ORB + PnP", v: 12.2, c: COLORS.accent, d: 315 },
              ].map((b) => (
                <div key={b.k}>
                  <div style={{ fontFamily: INTER, fontSize: 19, color: COLORS.offWhite, marginBottom: 6 }}>{b.k}</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ height: 26, width: 2.2 * bar(b.v, b.d), background: b.c, borderRadius: 5 }} />
                    <div style={{ fontFamily: INTER, fontWeight: 800, fontSize: 24, color: b.c }}>{bar(b.v, b.d).toFixed(1)}%</div>
                  </div>
                </div>
              ))}
            </div>
          </Step>
        </div>
      </AbsoluteFill>
      <Captions
        lines={[
          { from: 6, to: 140, text: "Every pixel step runs on OpenCV 5. YOLOX-s runs on the new DNN engine." },
          { from: 140, to: 282, text: "Look-closer defogs and zooms. StereoSGBM computes depth from the stereo pair." },
          { from: 282, to: 418, text: "Dense odometry failed on foliage. ORB features and solvePnPRansac track each hiker." },
        ]}
      />
    </SceneBg>
  );
};

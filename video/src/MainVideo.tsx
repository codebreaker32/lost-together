//    0 –  480  (0:00 – 0:16)  Hook: lost hikers, real YOLOX output in fog, detection range
//  480 –  870  (0:16 – 0:29)  Idea: one living map, look closer with 3x zoom
//  870 – 1620  (0:29 – 0:54)  Hero: real viewer capture, seed 45 SOLO 2/4 vs SHARED MAP 4/4
// 1620 – 2040  (0:54 – 1:08)  OpenCV 5 pipeline
// 2040 – 2640  (1:08 – 1:28)  Science: pre-registration, 159,800 episodes, 26.8% -> 42.4%, honesty
// 2640 – 3270  (1:28 – 1:49)  Agent: real trace, ranger console, honest scoreboard
// 3270 – 3570  (1:49 – 1:59)  AWS
// 3570 – 3900  (1:59 – 2:10)  Close
import React from "react";
import { AbsoluteFill, Series } from "remotion";
import { COLORS, SCENES } from "./constants";
import "./fonts";
import { Hook } from "./scenes/Hook";
import { Idea } from "./scenes/Idea";
import { Hero } from "./scenes/Hero";
import { Pipeline } from "./scenes/Pipeline";
import { Science } from "./scenes/Science";
import { Agent } from "./scenes/Agent";
import { Aws } from "./scenes/Aws";
import { Close } from "./scenes/Close";

export const MainVideo: React.FC = () => (
  <AbsoluteFill style={{ background: COLORS.bg }}>
    <Series>
      <Series.Sequence durationInFrames={SCENES.hook}><Hook /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.idea}><Idea /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.hero}><Hero /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.pipeline}><Pipeline /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.science}><Science /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.agent}><Agent /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.aws}><Aws /></Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.close}><Close /></Series.Sequence>
    </Series>
  </AbsoluteFill>
);

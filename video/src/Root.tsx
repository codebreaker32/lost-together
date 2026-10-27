import React from "react";
import { Composition, registerRoot } from "remotion";
import { MainVideo } from "./MainVideo";
import { FPS, H, SCENES, TOTAL, W } from "./constants";
import { Hook } from "./scenes/Hook";
import { Idea } from "./scenes/Idea";
import { Hero } from "./scenes/Hero";
import { Pipeline } from "./scenes/Pipeline";
import { Science } from "./scenes/Science";
import { Agent } from "./scenes/Agent";
import { Aws } from "./scenes/Aws";
import { Close } from "./scenes/Close";

const scenes = { Hook, Idea, Hero, Pipeline, Science, Agent, Aws, Close } as const;
const durations = [SCENES.hook, SCENES.idea, SCENES.hero, SCENES.pipeline, SCENES.science, SCENES.agent, SCENES.aws, SCENES.close];

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="MainVideo" component={MainVideo} durationInFrames={TOTAL} fps={FPS} width={W} height={H} />
    {Object.entries(scenes).map(([id, C], i) => (
      <Composition key={id} id={id} component={C} durationInFrames={durations[i]} fps={FPS} width={W} height={H} />
    ))}
  </>
);

registerRoot(RemotionRoot);

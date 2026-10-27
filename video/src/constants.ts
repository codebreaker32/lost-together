export const FPS = 30;
export const W = 1920;
export const H = 1080;

export const COLORS = {
  bg: "#070b09",
  bgCard: "#0f1612",
  bgCard2: "#141d18",
  border: "#22302a",
  accent: "#4ade80", // shared map / brand
  accentDim: "#166534",
  solo: "#f5b041", // solo hikers
  bear: "#f0714f", // predator (calm vermilion)
  info: "#7cc4ff",
  white: "#f4f6f5",
  offWhite: "#cfd8d3",
  muted: "#7d8a83",
};

// Scene durations in frames (sum = total). Timeline comment lives in MainVideo.tsx.
export const SCENES = {
  hook: 480,
  idea: 390,
  hero: 750,
  pipeline: 420,
  science: 600,
  agent: 630,
  aws: 300,
  close: 330,
};
export const TOTAL = Object.values(SCENES).reduce((a, b) => a + b, 0);

// playback rate of the real viewer capture inside the Hero scene (set after measuring the capture)
export const HERO_RATE = 1;

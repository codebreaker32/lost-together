# Lost Together demo video: narration script

The video (`video/out/lost-together-demo.mp4`, 2:10, 1920×1080, 30 fps, no audio track) is caption-driven. No text-to-speech was available on this machine: there is no ElevenLabs, Azure or OpenAI key in the environment, and piper and espeak-ng are not installed. The narration is therefore burned in as on-screen captions. The lines below are those captions word for word, with the times they appear.

**To add a voice-over:** record each line so it starts at its time code and ends before the next line starts. One continuous take is fine. Then mux it in from `video/`:

```bash
npx remotion ffmpeg -i out/lost-together-demo.mp4 -i voiceover/narration.wav -c:v copy -c:a aac -b:a 192k -map 0:v:0 -map 1:a:0 -shortest out/lost-together-demo-vo.mp4
```

**Tone:** calm, formal and plain. Say "intercepted" and "got out", never "killed" or "attacked". Any background music must be royalty-free or made by the team.

**Numbers:** every number comes from RESULTS.md or README.md (Tier A v1 calibrated; Tier C live run). Don't round them differently from the text below.

## 0:00.00–0:16.00 · Hook: lost, and the camera barely sees

| # | Start | End | Line |
|---|---|---|---|
| 01 | 0:00.20 | 0:02.40 | Four hikers are lost in a forest. |
| 02 | 0:02.40 | 0:05.07 | A predator is nearby. Each hiker has one camera. |
| 03 | 0:05.20 | 0:08.27 | At 10 metres, the detector finds the bear. In thick fog, it reports an elephant. |
| 04 | 0:08.27 | 0:11.00 | At 30 metres, it finds nothing. |
| 05 | 0:11.20 | 0:15.87 | Calibrated detection range: about 15 metres. |

## 0:16.00–0:29.00 · The idea: one living map, look closer

| # | Start | End | Line |
|---|---|---|---|
| 06 | 0:16.20 | 0:19.07 | Does sharing what each hiker sees help them get out? |
| 07 | 0:19.07 | 0:22.60 | Each camera view feeds one shared, living map. Old sightings fade. |
| 08 | 0:22.73 | 0:26.00 | A teammate's report tells a hiker where to look closer, with a 3× zoom. |
| 09 | 0:26.00 | 0:28.93 | Clear air: bear found at 20 metres. Light fog: missed. |

## 0:29.00–0:54.00 · Real viewer capture: seed 45, run twice

| # | Start | End | Line |
|---|---|---|---|
| 10 | 0:29.13 | 0:31.80 | Same forest, same predator, same random seed. Run twice. |
| 11 | 0:32.00 | 0:37.33 | Left: each hiker keeps a private map. Right: one shared map. |
| 12 | 0:37.33 | 0:43.00 | Sightings, exits and explored ground reach every hiker. Old evidence fades. |
| 13 | 0:43.00 | 0:49.33 | Only a separate evaluator holds ground truth. The hikers never see it. |
| 14 | 0:49.67 | 0:53.93 | Solo: 2 of 4 got out. Shared map: 4 of 4. A selected example, not the average. |

## 0:54.00–1:08.00 · OpenCV 5 pipeline

| # | Start | End | Line |
|---|---|---|---|
| 15 | 0:54.20 | 0:58.67 | Every pixel step runs on OpenCV 5. YOLOX-s runs on the new DNN engine. |
| 16 | 0:58.67 | 1:03.40 | Look-closer defogs and zooms. StereoSGBM computes depth from the stereo pair. |
| 17 | 1:03.40 | 1:07.93 | Dense odometry failed on foliage. ORB features and solvePnPRansac track each hiker. |

## 1:08.00–1:28.00 · Results

| # | Start | End | Line |
|---|---|---|---|
| 18 | 1:08.20 | 1:13.67 | Predictions were committed first. Then 159,800 calibrated episodes ran on AWS Graviton. |
| 19 | 1:13.87 | 1:18.00 | In clear air, sharing raised the escape rate from 27% to 42%. |
| 20 | 1:18.00 | 1:22.00 | Sharing beat solo in all 56 settings, at every fog level tested. |
| 21 | 1:22.20 | 1:27.93 | One prediction was partly wrong: shared sightings kept helping against a predator that senses the whole forest. |

## 1:28.00–1:49.00 · Agent and ranger

| # | Start | End | Line |
|---|---|---|---|
| 22 | 1:28.13 | 1:31.67 | On uncertain sightings, an LLM agent calls OpenCV tools. |
| 23 | 1:31.67 | 1:35.73 | It queries the map, checks for motion, and looks closer: bear 0.16 becomes bear 0.58. |
| 24 | 1:35.73 | 1:39.00 | It orders an escape. The evaluator confirms a real bear. |
| 25 | 1:39.13 | 1:43.20 | A human ranger approves every escape order. If the model fails, the rule decides. |
| 26 | 1:43.33 | 1:48.93 | The agent is not more accurate than the rule: fewer false alarms, more missed bears in thick fog. |

## 1:49.00–1:59.00 · AWS

| # | Start | End | Line |
|---|---|---|---|
| 27 | 1:49.13 | 1:54.00 | AWS: Graviton for sweeps, an L40S GPU for the agent, Lambda for the viewer. |
| 28 | 1:54.00 | 1:58.93 | GPU spend is capped at $10; about $2.60 used. GPU instances stop themselves. |

## 1:59.00–2:10.00 · Close

| # | Start | End | Line |
|---|---|---|---|
| 29 | 1:59.20 | 2:04.33 | Solo: 27% got out. Shared map: 42%. |
| 30 | 2:04.67 | 2:09.93 | Lost Together. All simulated. Every prediction reported. |

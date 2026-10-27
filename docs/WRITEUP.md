# Lost Together: write-up

*OpenCV AI Competition 2026, powered by AWS. All results are from simulation. Every other number comes from [RESULTS.md](../RESULTS.md) or a file in `results/`.*

## Inspiration

When a group is lost and a hazard is moving toward it (a wildfire front, rising water, a bear), each person knows only what they can see. Rescue apps and radios let people share what they see, but is sharing actually worth the effort? Could a noisy shared picture even make things worse, by spreading false alarms or by pulling everyone together into one easy-to-find target?

We wanted to answer that question with numbers rather than intuition. We framed it as a clean experiment: four lost hikers, one uncertain predator, and a "living map" that the hikers build from their own camera views. Every claim is checked against ground truth that only the evaluator holds. We wrote our predictions down before the main runs, so we could find out where we were wrong.

## What it does

- **Hikers see the forest through cameras.** We use real-looking frames from the TartanGround forest dataset. A bear, deer, a dog or other hikers are composited in with correct scale, occlusion and stereo disparity, and fog is added on top.
- **OpenCV 5 turns pixels into evidence:**
  - YOLOX-s on the new DNN engine finds candidates;
  - StereoSGBM gives depth;
  - RGB-D odometry, checked with ORB features and `solvePnPRansac`, tracks where each hiker is.
- **Each hiker keeps a map**: cells seen, exits found, and predator sightings with a confidence. Depending on the condition, hikers share nothing, only sightings, only terrain, or everything. A fifth condition also makes them walk as a group. The shared map is "living": old predator evidence fades.
- **A planner decides every step.** On uncertain events, an **LLM hiker agent** can spend look-again actions through an **OpenCV 5 MCP server**:
  - look closer, which defogs, enhances, zooms and re-detects;
  - compare with an earlier frame, using optical flow (animals move; rocks don't);
  - query the shared map.

  It then decides to keep going, look again, regroup or escape. An escape triggers an evacuation broadcast, which a **ranger** approves or overrides in `advise` mode.
- **An evaluator scores everything** against ground truth: who got out, how close the predator came, false-alarm escapes and late escapes.
- **A replay viewer** shows the same forest seed twice, side by side: alone vs. with a shared map. In the demo seed (45, calibrated perception, light fog), 2 of 4 hikers get out alone and 4 of 4 get out with the shared map. That seed was picked by the exporter as a clear contrast; the average gain across the grid is +15 points at that fog level.

## How we built it

- **Simulator (Python, NumPy).**
  - The world is an 80×80-cell forest: 5 m cells, 4-second steps, 40-minute episodes.
  - The predator pursues the nearest hiker it senses. A robustness variant waits in ambush near an exit.
  - Named random streams per seed make every condition see the same forest, spawn points and predator start (common random numbers).
  - Hikers receive only `Observation`s, and a unit test fails if hiker code touches the world.
- **Vision (OpenCV 5.0.0, pip `opencv-python==5.0.0.93`).**
  - `cv2.dnn.readNetFromONNX(..., ENGINE_NEW)` runs YOLOX-s from the OpenCV Zoo, decoded with grid/stride and `NMSBoxes`.
  - `StereoSGBM` runs in 3-way mode.
  - `ptcloud` `Odometry` runs with `OdometryType_RGB_DEPTH`, gated by ORB + `solvePnPRansac`.
  - `grabCut` cut animals and hikers out of licensed photos.
  - The compositor places each object at a metric distance using the pinhole model and hides it per pixel behind nearer scene depth. In the right view, it shifts the object by the disparity f·B/Z. Fog follows the scattering model.
- **Agent.** One OpenCV 5 tool layer is served over MCP (stdio). The model is `qwen3-vl:30b-a3b-instruct` on Ollama on an EC2 g6e.xlarge (NVIDIA L40S), called through an OpenAI-compatible client with retries, timeouts and a request-hash cache. The cache's live/replay/stub modes let anyone rerun the benchmark offline. The agent may use at most 3 tool turns and must answer in strict JSON. Every failure falls back to the rule decision and records why.
- **Experiment.** Tier A is a phase diagram over predator speed × sensing radius, with five cooperation conditions and 200 paired seeds per cell. It adds a predator-off control and a perfect-perception reference. Tier B (full pixels) and Tier C (rule vs. LLM on identical decision points, McNemar's test) follow.
- **AWS (ap-south-1):**
  - Sweeps: one Graviton **c8g.2xlarge** at about $0.22/h, about 60 episodes/s. A 56,000-episode sweep took 743 s, about $0.05.
  - The agent: a GPU account with a **$10 cap** tracked by AWS Budgets alerts and held by stop-after-batch discipline and self-stopping instances.
  - Results and the viewer: a private **S3** bucket, defined in the site template and not deployed yet.
  - Hosting template: **CloudFormation + CloudFront**, blocked pending account verification.
  - Metrics: a **CloudWatch** metrics publisher and dashboard (built; dry run by default).
  - Access: a **least-privilege IAM** developer user on the GPU account (the default account still uses root keys; fixing that is on the list).
- **Development.** The team set the question, the scenario and the constraints. AI coding agents (Claude Code) helped write and review much of the code, the experiment design and these documents, under the team's direction.

## Challenges

- **There is no GPU simulator.** A live Unreal forest needs a GPU we could not get for long runs. We built a cheap grid-world simulator for the 10⁵-episode sweeps instead, and plan to calibrate its perception model from the real OpenCV pipeline.
- **Our first "sharing helps" result was mostly navigation.** In the pilots, most of the benefit of sharing came from finding exits faster. With the predator off, full sharing still raised escapes from 61.4% to 90.5% (v0). So we split sharing into `sightings` and `map` and added a predator-off control. This was the most important design change we made.
- **Dense odometry failed on foliage.** On 121 forest frames with ground-truth depth, `ptcloud` RGB-D odometry alone drifted 88.5% of the path length. ORB + `solvePnPRansac` drifted 12.2%, and 20.3% on our own SGBM stereo depth (15–28% in fog and with sensor noise). The odometry call reported success while returning a translation of about 10⁻⁴ m for a 0.42 m step, so a success flag is not a validity check, and every estimate is now cross-checked.
- **Small animals in leaves are hard for a COCO detector.** In one example frame, YOLOX finds the bear at 10 m (0.82) but not at 30 m. In fog it calls it an "elephant". This is why the agent has `look_closer` and why a classical dark-blob pass sends "unknown" shapes to the agent instead of dropping them.
- **VLMs are poor at pixel coordinates.** Our tools take OpenCV candidate IDs instead: OpenCV finds the regions, and the model chooses among them.
- **Accounts and quotas.** Bedrock was unavailable, GPU Spot quota was 0, and CloudFront needs account verification. We self-hosted the model on EC2, split the work across two accounts, and wrote a fallback hosting path.

## Accomplishments that we're proud of

- **Pre-registered predictions with an honest scorecard** (SHA-256 `790902a3…`). Three of the five predictions hold, one partly holds, and one (H3) is **partly falsified**: we predicted that shared sightings would stop helping when the predator senses the whole forest, and they don't.
- **A clear effect with a control behind it, on measured perception.** With the detection model calibrated from YOLOX on fogged composites (Tier A v1), sharing raised the escape rate from 26.8% to 42.4% across the grid, with all 56 cells' CIs above 0 at each of three fog levels. Sharing a map built from that imperfect vision recovered 56% of the gap to perfect perception (the v0 dress rehearsal, with a guessed noise model, had said 81%).
- **An agent that fails safely.** Timeouts, invalid JSON, unknown tools, bad arguments, endless tool loops and model-call errors all end in the rule decision. A test covers each of these failures.
- **A human in control.** Evacuation broadcasts wait for a ranger in `advise` mode, and every approval or override is logged.
- **Everything is reproducible:**
  - pinned dependencies;
  - per-sweep argument files;
  - paired seeds;
  - a test suite that covers every agent fallback path;
  - a replay mode that reruns the agent benchmark without a GPU.
- **Cheap to run.** A full 56,000-episode phase-diagram sweep costs about five cents on Graviton.

## What we learned

- **Sharing helps more broadly than we expected, but for two different reasons.** Shared terrain helps everyone find exits. Shared *sightings* help once the predator is about as fast as the hikers or faster (in 5 of 7 cells at 1× speed, 2 of 7 at 0.75×, none at 0.5×), and they kept helping even when the predator could sense the whole forest (v0 and v1 agree). Fog erodes the sightings channel (+0.059 → +0.034 at 39 m visibility) but not the terrain channel.
- **Moving as a group was worse than sharing while staying apart.** `group` scored below `share` in 44 of 56 cells and above it in none (v1; mean −0.073). Groups are easier to sense and lose time gathering.
- **A shared map is only as good as the radio.** In the v0 robustness sweep (guessed perception), limiting sharing to 50 m cut the benefit from +0.12…+0.27 to +0.05…+0.10.
- **Controls matter more than clever models.** The predator-off control changed how we read the whole result.
- **In OpenCV 5, check before you trust.** API names moved between 4.x and 5.0, and the Python odometry sample is broken. Odometry needs float32 intrinsics and an explicit max depth for outdoor scenes. Its success flag is not a quality check.

## What's next

- **Denser fog:** v1 found no visibility down to 39 m at which sharing stops helping (the benefit shrinks from +0.156 to +0.104). Extend the fog axis past β = 0.1 to find the boundary.
- **Tier B:** full-pixel runs at three anchor cells, to check that the surrogate agrees with pixels.
- **Tier C, fog robustness:** the live agent (`qwen3-vl:30b-a3b-instruct`, 360 points, $0.0026 per decision) matched the rule on accuracy (0.606 vs 0.578, McNemar p = 0.39). It halved false alarms (57 vs 110) but missed more real bears (85 vs 42 late escapes), most of them in thick fog. Next: give it a fog-aware prior, or escalate to a human when visibility is low.
- **Localisation:** odometry on stereo depth and in fog, plus pose-graph fusion with GTSAM. Then a real-forest check on FoMo or FinnForest.
- **COOL:** run the per-frame pipeline on the COOL Graviton AMI and compare cost per 1,000 hiker-frames against the stock wheel. Not part of this submission: it needs a paid Marketplace subscription.
- **Hosting:** a public HTTPS replay and results site once AWS verifies the account: {{TBD: URL}}.

## Built with

- **OpenCV 5.0.0:**
  - `dnn` (new engine);
  - `StereoSGBM`;
  - `ptcloud` `Odometry`;
  - `features` (ORB);
  - `solvePnPRansac`;
  - `calcOpticalFlowFarneback`;
  - `CLAHE`;
  - `grabCut`.
- **OpenCV Zoo:** YOLOX-s.
- **Languages and libraries:**
  - Python 3.12, NumPy, SciPy, Matplotlib;
  - MCP Python SDK;
  - uv;
  - vanilla JavaScript (replay viewer).
- **Model:** Ollama serving `qwen3-vl:30b-a3b-instruct`.
- **AWS:**
  - EC2 (Graviton c8g, g6e with an L40S GPU);
  - S3;
  - CloudFormation, CloudFront;
  - CloudWatch;
  - IAM;
  - AWS Budgets.
- **Data:** TartanGround (CC BY 4.0), plus Wikimedia Commons photos (public domain, CC0, CC BY 2.0).

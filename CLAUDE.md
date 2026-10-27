# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is the team's project for the **OpenCV AI Competition 2026 (powered by AWS)** — https://opencv26.devpost.com/. No code exists yet. The agreed concept, "Lost Together", is in [PROJECT_BRIEF.md](PROJECT_BRIEF.md): simulated hikers lost in a forest share a living map built from their own camera views, while an uncertain predator hunts them. The research question is whether working together helps. Read the brief before designing anything.

Constraints the team set: no self-collected footage (use datasets or simulation only); the forest/predator/hiker concept is fixed; agents never read simulator ground truth (only a separate evaluator does). Only commit when the user asks.

## Commands

```bash
uv sync                                                    # create .venv from pyproject/uv.lock
uv run pytest -q                                           # all tests
uv run pytest tests/test_sim.py::test_same_seed_same_result -q   # a single test
uv run python -m lost_together.sweep --name pilot --speed-ratio 0.5 1 1.5 --sense-radius 10 20 30 --coop solo share group --seeds 30
uv run python -m lost_together.analysis results/pilot.csv  # phase-diagram PNGs and a Δ table next to the CSV
```

Sweeps write `results/<name>.csv` (one row per episode) and `results/<name>.json` (the exact args and base config). `results/` is gitignored. Findings so far are in RESULTS.md; predictions are in PREREGISTRATION.md, which must not be edited after main runs start.

**Big sweeps run on AWS:** `tools/sweepbox.sh up|sync|run NAME ARGS|status|fetch|down`. This uses one c8g.2xlarge on the default account at ~$0.22/h. It does about 60 episodes/s, so 56k episodes take ~15 min. Run `sync` after code changes, and run `down` when done.

```bash
uv run python -m lost_together.figures results/tierA_v0.csv --control results/tierA_v0_off.csv   # headline figure
uv run python -m lost_together.export --find-contrast 60 --coops solo sightings map share group --left solo --right share --out viewer/data
python -m http.server -d viewer 8000                       # replay viewer (hero seed 26: solo 1/4 out, share 4/4)
uv run python -m lost_together.escalations --seeds 2000 2060 --out results/escalations_raw.jsonl   # Tier C decision points
```

**Hosting:** the default account can't create CloudFront distributions or Amplify apps until AWS Support verifies it. Error ID b2d96f82-d319-417b-938d-3c91ac7c8054.
- After verification, use `infra/aws/deploy_site.sh` (private S3 bucket + CloudFront + HTTPS).
- The interim option is `infra/aws/deploy_s3_website.sh`, a public HTTP S3 website. The team must run it themselves, because Claude's permission settings block making a bucket public.
- `infra/aws/deploy_amplify.sh` is a third option.

## Code layout (L0 simulator, `lost_together/`)

- **Ground-truth side:** `world.py` (forest, exits, line of sight, `distance_field` shortest paths), `predator.py`, `sim.py` (episode loop and metrics) and the truth log in `perception.py`.
- **Hiker side:** `planner.py` (rule-based policy) and `knowledge.py` (the living map, alarm and risk heatmap). These may only see `Observation`s. `tests/test_sim.py::test_hiker_policy_never_touches_ground_truth` enforces this; keep it passing.
- **Perception:** `perception.py` is the only path from truth to hikers. The `SURROGATE` mode is the noise model that the real OpenCV pipeline will calibrate (Tier A); `PERFECT` is the reference.
- **Common random numbers:** `rng.py` gives named streams per seed, so the forest, spawn points and predator start are identical across cooperation conditions. Always compare conditions paired by seed.
- **Defaults:** in `config.py`. A cell is 5 m and a step is 4 s, so the default 80×80 grid is 400 m and T_max = 600 steps is 40 min.

## Competition context

The **official rubrics** (from the Devpost Rules page) are summarised in PROJECT_BRIEF.md §7:
- **Overall:** technical execution 30% (depth of the OpenCV 5 implementation and evaluation), innovation 20%, impact 20%, UX 10%, documentation 10%, cloud delivery, reproducibility and responsible operation 10%.
- **Awards stack:** one Overall Award plus one or both Special Awards.
- **COOL award:** requires *verified* COOL integration on Graviton.
- **Agentic award:** scores failure handling and human control.
- **Rejection grounds:** judges can reject gratuitously violent content, data used without rights, and misrepresented benchmarks.

The rest of this section comes from secondary sources. Devpost blocks automated fetches, so don't try to fetch it.

- **Brief:** build a system that uses **OpenCV 5** and **AWS** to "see, reason, and act". Visual understanding must drive a decision, action, prediction, or interaction. Physical AI and generative-AI systems are emphasized.
- **Prizes:** $5k / $3k / $2k for the top three places, plus two $1k special awards (one per track).
- **Any language or hardware is allowed.** Projects built with agentic tools such as Claude Code, Codex, Kiro, or MCP-compatible systems are reportedly viewed favorably.

### Tracks (enter one, both, or neither)

1. **Cloud-Optimized Vision with COOL** (Best Use of Cloud-Optimized OpenCV Library award). COOL is available through AWS Marketplace and optimized for AWS Graviton. Strong entries document a **reproducible deployment** and a **comparison against an appropriate baseline**, so record benchmark numbers as you build rather than at the end.
2. **Agentic Vision with OpenCV 5** (Agentic Vision award). An agent calls OpenCV 5 tools in a multi-step perception → decision → action loop. The visual evidence must change what the system does next: a plan, a tool call, an action, or a request for human approval. **A chatbot that only explains a fixed vision result does not qualify.**

Suggested domains: healthcare, safety, accessibility, agriculture, environmental monitoring, smart cities, education, retail, and sports analytics. Organizers called out active perception for inspection, physics-informed video prediction, multi-agent visual SLAM, and real-time spatial digital twins.

## Technical notes

- **OpenCV 5.0.0** (released 2026-06-06; pip `opencv-python==5.0.0.93`) runs modern models through its own DNN engine, with no PyTorch or ONNX Runtime required. The new engine is CPU-only. Its API differs from 4.x: `calib3d` is split into `geometry`, `calib` and `stereo`; `features2d` is renamed `features` (it adds ALIKED, DISK, LightGlueMatcher and ANNIndex; you supply the ONNX weights); RGB-D odometry and TSDF live in `ptcloud`. Python calls stay at `cv2.<func>`. Check 5.0.0 headers rather than assuming 4.x signatures.
- **OpenCV 5.0.0 SLAM gaps:** there is no bundle adjustment, loop closure or place recognition, and `PoseGraph`/`LevMarq` are C++-only. Use GTSAM for pose graphs. `ptcloud` Odometry has no RGB-only mode, so it needs depth (stereo or rendered). `cv::slam::VisualOdometry` exists only on the unreleased 5.x branch; don't depend on it. The 5.0.0 Python odometry sample is broken: use `cv.OdometryType_DEPTH`.
- **Verified on the 5.0.0.93 wheel:**
  - DNN engine constants are `cv2.dnn.ENGINE_CLASSIC=1`, `ENGINE_NEW=2`, `ENGINE_AUTO=3` and `ENGINE_ORT=4`; select one with `cv2.dnn.readNetFromONNX(path, engine)`.
  - OpenCV Zoo YOLOX (`object_detection_yolox_2022nov.onnx`, Apache-2.0) gives identical detections on NEW and CLASSIC. Its output is (1, 8400, 85), decoded with grid/stride (8/16/32) and `cv2.dnn.NMSBoxes`. Pre-processing is BGR→RGB, letterbox to 640 padded with 114, raw 0–255 input.
  - These Python APIs exist: `StereoSGBM_create`, `Odometry`, `OdometryType_RGB_DEPTH`, `OdometryFrame`, `Volume`, `ALIKED_create(onnx)`, `LightGlueMatcher_create(onnx)`, `ANNIndex_create(dim)`, `solvePnPRansac`.
  - Check script: `tools/yolox_check.py MODEL.onnx IMAGE...`. It needs `opencv-python==5.0.0.93`, which isn't yet a project dependency; add it when perception work starts.
- **Licences:** keep GPL SLAM code (ORB-SLAM3, COVINS, g2o's GPL parts) out of the submission. Don't use WildfireX-SLAM: its licence bans use with generative AI.
- **COOL** runs only on Graviton EC2 (Marketplace AMI) or ECS/EKS (container), never on Lambda or Fargate. Launch from the Marketplace AMI ID, not from copied AMIs. The stock aarch64 wheel already includes KleidiCV, so measure before claiming speedups.
- **Two AWS accounts, split by the team's rule:**
  - **GPU only:** account …0916, through profile `lost-together` (IAM user `lost-together-dev`, scoped by `infra/iam/`). Its root-key profile `bhawna` is for admin tasks only.
  - **Everything else** (COOL, S3, SQS, CloudWatch, CDK, the sweep, the replay site): the **default** CLI profile, account …8661. The AWS MCP tools point at this account too. It still uses root keys, so create a scoped user with `bash infra/iam/create-dev-user.sh default <new-profile>`.

  Region for both accounts is **ap-south-1**. COOL is listed there: Graviton AMI `ami-019a34dc69ce62cb0`, "OpenCV-graviton-optimized-AMI-prod-…".
- **Quotas** (queried 2026-10-26):

  | Quota | GPU account …0916, ap-south-1 | Default account …8661, ap-south-1 and us-east-1 |
  |---|---|---|
  | EC2 on-demand Standard | 16 vCPU | 8 vCPU |
  | EC2 on-demand G/VT | **8 vCPU** | 0 |
  | EC2 Spot G/VT | 0 | — |
  | SageMaker GPU endpoints | 0 | 0 |
  | Bedrock | tokens/day = 0 | denied |
- **Cross-account traffic:** the agent's model server (Ollama on the GPU account) must not be open to the internet. Allow only the default account's caller, via security-group source IP or an SSH/SSM tunnel. Alternatively, run GPU batches on the GPU box reading inputs from S3 through a cross-account bucket policy.
- **Standing rule from the user: never leave billable resources running when they aren't in use.**
  - The moment a job finishes or a box goes idle (e.g. waiting on another workstream), run `tools/sweepbox.sh down` or `tools/gpubox.sh down`, delete databases and endpoints, and release Elastic IPs. Relaunching takes about 3 minutes, which is always cheaper than idling.
  - Before ending a turn, audit both accounts (`default` and `lost-together`, ap-south-1 and us-east-1) for running instances.
  - Per-request services such as the Lambda viewer can stay up. Leave resources the team created themselves alone unless asked; for example, the default account has a pre-existing unattached EIP in ap-south-1 and a stopped m7i-flex.large in us-east-1.
- **Free Tier credits are claimed by `infra/credits/free-tier-activities.sh`.** Activities done through the API do count.
- **GPU spend is capped at $10 in total.** The AWS Budget `lost-together-gpu-cap` (annual, gross of credits) alerts smrtamy23@gmail.com at 50/80/100%.
  - **GPU:** g6e.xlarge (L40S, 48 GB) in ap-south-1 at $2.235/h, about **4.5 h in total**. The fallback for longer runs is g6.xlarge (24 GB) at $0.966/h.
  - **Sessions:** start the instance only for batched benchmark or demo runs, and stop it as soon as the batch finishes.
  - **Dead-man switch:** launch with `--instance-initiated-shutdown-behavior stop` and `shutdown -h +90` in user-data, so a forgotten instance stops by itself.
- **Agent model:** **`qwen3-vl:30b` with thinking off** (vision + tools, Apache-2.0), served by **Ollama** on the g6e.xlarge. GPU smoke test on 2026-10-26:
  - **Use the tag `qwen3-vl:30b-a3b-instruct`.** On Ollama 0.40.11 (2026-10-26 evening) the bare `qwen3-vl:30b` tag is the thinking variant and ignores both `reasoning_effort: "none"` and `think: false`, so every answer came back empty.

  | | `qwen3-vl:30b` | `qwen3-vl:8b` |
  |---|---|---|
  | Seconds per step (thinking off) | **~2.1** | ~4.7 |
  | Decode speed | 182 tok/s | 116 tok/s |
  | First step with thinking on | 16 s | 36 s |
  | VRAM | 22 GB | 10 GB |

  - Both models completed the look-again loop in every message format, including images inside tool messages in both the OpenAI-compatible and native APIs.
  - Both models fit on the L40S together (35 of 46 GB used).
  - Pull time on AWS: 8b in 30 s, 30b in 150 s.
  - Thinking adds latency without changing the decision: in the OpenAI-compatible API send `reasoning_effort: "none"`; in the native API send `think: false`.
  - The test images were synthetic and easy, so this doesn't measure accuracy on real frames.
- **No local model development.** On this machine (CPU only, 10 GB RAM), `qwen3-vl:4b` took 498 s per step and returned nothing. Develop agent logic locally against recorded or stubbed responses, and make real calls only in GPU sessions.
- **Groq** (`qwen/qwen3.8-27b`, `GROQ_KEY` in `.env`; never print it or commit `.env`) is a fallback only. It rejects images in tool messages.
- **GPU launch user-data:** cloud-init has no `$HOME`, and the Ollama CLI panics without it. So:
  - `export HOME=/root`;
  - `systemctl enable --now ollama`;
  - wait for `:11434`, then pull;
  - `shutdown -h +45` with `--instance-initiated-shutdown-behavior terminate`.

  Allow SSH only from the team IP, and reach Ollama through an SSH tunnel; never expose port 11434.
- **Agent tool rules:**
  - Tools take OpenCV detector candidate IDs, not x/y coordinates, because VLMs are poor at pixel coordinates.
  - Keep a fallback that puts images in a follow-up user message, so providers stay swappable.
  - Cache responses by input hash.
- **Simulator:** a viewpoint graph over TartanGround/TartanAir frames, with the predator, decoys and hikers composited in using ground-truth depth and fog added via the scattering model. Perception is cached per (node, view, fog). The design is in PROJECT_BRIEF.md §4. Don't pip-install anything that pulls `opencv-python` on the COOL box, because it shadows COOL's `cv2`.
- **AWS work:** use the `aws-mcp` tools (`run_script` and others), not the AWS CLI through Bash. For Bedrock, AgentCore, CDK, and similar services, load the matching `aws-*` / `amazon-bedrock` skill first.
- **Submission assets:** the `hackathon-submission` skill (pitch deck and judge review) and the `demo-video` skill (Remotion demo video) are available for the final submission.

# AWS delivery: replay site and experiment metrics

This folder puts the Lost Together **replay viewer** and **raw results** on a public HTTPS URL that judges can open without an AWS account. It also publishes each sweep's headline numbers to a **CloudWatch dashboard**. Everything runs in the team's default account (…8661), in **ap-south-1**.

```
viewer/  ──aws s3 sync──▶  S3 bucket (private, SSE-S3)  ◀──SigV4 (OAC)──  CloudFront  ──HTTPS──▶  judges
results/*.csv|json|png ──▶  s3://…/results/                                  (PriceClass_200, incl. India)
results/<sweep>.csv ──publish_metrics.py──▶ CloudWatch "LostTogether" metrics ──▶ dashboard "LostTogether"
```

| File | Purpose |
|---|---|
| `site.yaml` | CloudFormation: bucket, Origin Access Control, response-headers policy, distribution, bucket policy |
| `deploy_site.sh` | Deploy or update the stack, sync `viewer/` and `results/`, invalidate CloudFront, print the URL |
| `destroy_site.sh` | Empty the bucket and delete the stack |
| `publish_metrics.py` | Summarise a sweep CSV, send `put-metric-data` and create the dashboard. Dry run by default. |
| `dashboard.json` | Dashboard layout: experiment bars, plus CloudFront traffic and error rates |

## What gets created

One CloudFormation stack, `lost-together-site`. Every taggable resource is tagged `project=lost-together` and `component=replay-site`.

| Resource | Settings |
|---|---|
| `AWS::S3::Bucket` `lost-together-site-<account>-ap-south-1` | All four Block Public Access flags on; ACLs disabled (`BucketOwnerEnforced`); SSE-S3 (AES-256) with Bucket Key; abort incomplete multipart uploads after 1 day; no versioning, so teardown is clean |
| `AWS::CloudFront::OriginAccessControl` | SigV4, always signs; the S3 origin uses OAC, not the legacy OAI |
| `AWS::CloudFront::ResponseHeadersPolicy` | HSTS (2 years, includeSubDomains), `Content-Security-Policy` (a stack parameter), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `X-XSS-Protection: 0` (per OWASP; CSP does the real work), `Permissions-Policy` (camera, microphone, geolocation, payment and USB off), `Cross-Origin-Opener-Policy: same-origin`; removes `Server` |
| `AWS::CloudFront::Distribution` | HTTP→HTTPS redirect, `index.html` as default root object, compression, HTTP/2 and HTTP/3, IPv6, `PriceClass_200`, managed `CachingOptimized` policy, GET/HEAD/OPTIONS only, 403/404 → `/index.html` with status 404 |
| `AWS::S3::BucketPolicy` | `s3:GetObject` only for `cloudfront.amazonaws.com` with `AWS:SourceArn` = this distribution; denies every request that does not use TLS |

The stack creates **no IAM roles, compute or databases**, and no other resources.

`publish_metrics.py --apply` additionally writes custom metrics to the `LostTogether` namespace and creates or updates one dashboard, `LostTogether`. Neither belongs to the stack (see teardown below).

## Rough monthly cost

| Item | Estimate |
|---|---|
| S3 Standard storage (viewer + results, < 100 MB) | < $0.01 |
| CloudFront (always-free tier: 1 TB out and 10 M requests per month) | $0 at judging-scale traffic |
| Invalidations (first 1,000 paths per month free; each deploy uses one `/*` path) | $0 |
| CloudWatch custom metrics: about 30–50 per sweep at $0.30 per metric-month, prorated by the hour in which a metric is sent. Publishing once uses about one hour. The first 10 metrics are free. | ≈ $0.01–0.05 per sweep published |
| CloudWatch dashboard (first 3 per account free, then $3/month) | $0–3 |
| CloudFormation | $0 |
| **Total** | **≈ $0–$1/month, under $4 even with no free tier** |

## Deploy

Prerequisites: AWS CLI v2, credentials for the default profile (account …8661), and a built `viewer/index.html`.

```bash
bash infra/aws/deploy_site.sh                 # prints https://dXXXX.cloudfront.net/
```

- The **first** deploy takes about 3–10 minutes while the distribution is created. After that, a re-run updates the stack only if `site.yaml` changed, uploads only changed files, deletes removed files and invalidates `/*`.
- Each file kind gets an explicit `Content-Type` with charset. Without this, the CLI's guesser gets `.mjs`, `.wasm` and `.webmanifest` wrong.
- Cache lifetimes: HTML gets `no-cache`; JSON, CSV, Markdown, JS and CSS 5 minutes (they are not fingerprinted, so a browser must not keep an old `app.js` after `index.html` changes); images, video and fonts 1 hour. The invalidation flushes CloudFront on every deploy.
- `results/*.csv|json|png|svg|md` are published under `/results/` as the public read-only results path for judges. Logs and dotfiles are never uploaded. Set `INCLUDE_RESULTS=0` to skip this.
- The script **refuses to run in any account not ending in 8661**. This keeps the site out of the GPU account.

Environment overrides: `AWS_PROFILE` (default `default`), `AWS_REGION` (`ap-south-1`), `STACK_NAME` (`lost-together-site`), `VIEWER_DIR`, `RESULTS_DIR`, `INCLUDE_RESULTS`, `EXPECTED_ACCOUNT_SUFFIX` (set it to `""` to disable the check), `WAIT_FOR_INVALIDATION=1`, `CSP` (a new Content-Security-Policy).

The CSP allows this site, inline script and style, and the pinned CDNs `cdnjs`, `jsdelivr`, `unpkg` and Google Fonts. If the viewer needs something else, or to tighten it once the viewer is final, run `CSP="default-src 'self'; ..." bash infra/aws/deploy_site.sh`. Editing the default in `site.yaml` alone does not change a stack that already exists, because `cloudformation deploy` keeps the previous value of any parameter it is not given.

## Metrics and dashboard

```bash
uv run python infra/aws/publish_metrics.py results/pilot3.csv            # DRY RUN (default): table + payloads
uv run python infra/aws/publish_metrics.py results/pilot3.csv --apply    # put-metric-data + put-dashboard
```

The script uses only the standard library and calls the AWS CLI, with no boto3, so it also runs on the COOL Graviton box. Each published point has the dimensions `Sweep`, `Perception`, `Predator` (on/off) and `Cooperation`:

| Metric | Meaning |
|---|---|
| `EscapeRate` | mean share of hikers who escaped (%) |
| `AllEscapedRate` | % of episodes in which every hiker escaped |
| `DeltaVsSolo`, `DeltaVsSoloCILow`, `DeltaVsSoloCIHigh` | difference from solo in percentage points, paired by seed (same forest, spawns and predator start), with a 95% bootstrap CI |
| `CellsWhereModeHelps`, `CellsWhereModeHurts`, `Cells` | phase-diagram cells (speed × sense radius × noise) whose per-cell CI lies above or below 0 |
| `FalseAlarmsPerEpisode`, `FalseAlarms` | false escape decisions, a measure of the agent's decision quality |
| `CaughtPerEpisode` | hikers intercepted per episode |
| `Episodes`, `TotalEpisodes` (per sweep) | sample sizes |

The dashboard widgets use `SEARCH()` expressions, so every published sweep shows up without editing the dashboard. If the site stack exists, `--apply` reads `DistributionId` from it and adds CloudFront widgets (requests, bytes, 4xx/5xx and total error rate, from us-east-1). `--apply` also checks the caller account suffix.

## Security and responsible operation

- **No public bucket.** Block Public Access is on and ACLs are disabled. Only this one distribution can read objects (`AWS:SourceArn`), it can only read them, and plain-HTTP requests to S3 are denied.
- **HTTPS only**, with HSTS and a restrictive CSP, framing denied and device APIs turned off. The site is static, with no forms, cookies or user data.
- **No IAM roles and no compute** in the stack. Nothing runs server-side, so it has no attack surface beyond S3 and CloudFront.
- **No secrets.** The scripts use the caller's CLI profile. Nothing is stored in the repo or in the template, and the sync never uploads dotfiles, `.env`, keys or logs.
- **Account guard.** All scripts refuse to act outside the expected account.
- **Dry run by default** for metrics. Teardown asks before it deletes.
- **Honest data.** Every published number is simulated. The `Perception` dimension says whether it came from the surrogate or the pixel pipeline; name agent sweeps so the tier is clear (for example `tierC-agent-...`). Dashboard bars average every publish of a sweep name in the last 14 days, so after fixing a bug, republish under a new `--sweep` name rather than the old one.

## Tear down

```bash
bash infra/aws/destroy_site.sh            # asks you to type 'delete'
bash infra/aws/destroy_site.sh --yes      # non-interactive
aws cloudwatch delete-dashboards --dashboard-names LostTogether --region ap-south-1   # optional
```

Deleting the stack disables and then deletes the distribution, which takes 5–15 minutes. Custom metrics cannot be deleted; they stop costing anything once nothing is sent and expire under CloudWatch retention.

## Known limits

- The site uses the default `*.cloudfront.net` certificate, so CloudFront allows TLS 1.0 as a minimum for older clients. A custom domain with ACM would allow `TLSv1.2_2021`.
- CloudFront access logs are off: standard logs need an ACL-enabled bucket. Traffic and error rates come from CloudFront's free CloudWatch metrics instead.
- No WAF. The site is static and read-only, and a WAF would add about $6/month.

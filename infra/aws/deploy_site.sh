#!/usr/bin/env bash
# Deploy (or update) the Lost Together replay site: CloudFormation stack -> S3 sync -> CloudFront invalidation.
#
#   bash infra/aws/deploy_site.sh                 # default profile, ap-south-1
#   AWS_PROFILE=other AWS_REGION=ap-south-1 bash infra/aws/deploy_site.sh
#   INCLUDE_RESULTS=0 bash infra/aws/deploy_site.sh   # viewer only, no results/ upload
#
# Idempotent: re-running with no changes is a no-op deploy, a sync that uploads nothing and one
# "/*" invalidation (the first 1,000 invalidation paths per month are free).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-ap-south-1}"
STACK="${STACK_NAME:-lost-together-site}"
PROJECT="${PROJECT_TAG:-lost-together}"
VIEWER_DIR="${VIEWER_DIR:-$REPO/viewer}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/results}"
INCLUDE_RESULTS="${INCLUDE_RESULTS:-1}"
# Team rule: the site lives in the default account (...8661), never the GPU account. Set to "" to skip.
EXPECTED_ACCOUNT_SUFFIX="${EXPECTED_ACCOUNT_SUFFIX-8661}"
WAIT_FOR_INVALIDATION="${WAIT_FOR_INVALIDATION:-0}"
# `cloudformation deploy` keeps the previous value of any parameter not overridden, so editing the CSP
# default in site.yaml does NOT change an existing stack. Set CSP="..." here to change it.
CSP="${CSP:-}"

aws_() { aws --profile "$PROFILE" --region "$REGION" "$@"; }
log() { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v aws >/dev/null || die "aws CLI v2 not found"
[[ -f "$VIEWER_DIR/index.html" ]] || die "no $VIEWER_DIR/index.html - build the viewer first (or set VIEWER_DIR)"

# Guard against the wrong account.
ACCOUNT="$(aws_ sts get-caller-identity --query Account --output text)"
if [[ -n "$EXPECTED_ACCOUNT_SUFFIX" && "$ACCOUNT" != *"$EXPECTED_ACCOUNT_SUFFIX" ]]; then
  die "profile '$PROFILE' is account ...${ACCOUNT: -4}, expected ...$EXPECTED_ACCOUNT_SUFFIX (override EXPECTED_ACCOUNT_SUFFIX)"
fi
log "account ...${ACCOUNT: -4}, region $REGION, stack $STACK"

log "deploying CloudFormation stack (CloudFront changes can take 3-10 min on first create)"
aws_ cloudformation deploy \
  --stack-name "$STACK" \
  --template-file "$HERE/site.yaml" \
  --parameter-overrides "Project=$PROJECT" ${CSP:+"ContentSecurityPolicy=$CSP"} \
  --tags "project=$PROJECT" "component=replay-site" \
  --no-fail-on-empty-changeset

out() { aws_ cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
BUCKET="$(out BucketName)"
DIST_ID="$(out DistributionId)"
URL="$(out SiteURL)"
[[ -n "$BUCKET" && "$BUCKET" != "None" ]] || die "stack has no BucketName output"

# --- upload --------------------------------------------------------------------------------------
# Explicit Content-Type / Cache-Control per file kind (the CLI's guesser misses .mjs/.wasm/.webmanifest
# and adds no charset). HTML is revalidated on every load; data, JS and CSS (not fingerprinted, so they
# must not outlive index.html in browsers) are cached 5 minutes; images, video and fonts 1 hour.
# Every deploy invalidates CloudFront, so these lifetimes only matter for browser caches.
NOCACHE="no-cache, max-age=0, must-revalidate"
SHORT="public, max-age=300"
LONG="public, max-age=3600"

typed_sync() { # typed_sync SRC DEST CACHE CTYPE PATTERN...
  local src="$1" dest="$2" cache="$3" ctype="$4"; shift 4
  local inc=() found=0
  for p in "$@"; do
    inc+=(--include "$p")
    [[ -n "$(find "$src" -type f -name "$p" -not -path '*/.*' -print -quit)" ]] && found=1
  done
  ((found)) || return 0 # nothing of this kind locally: skip the bucket listing
  aws_ s3 sync "$src" "$dest" --only-show-errors --exclude "*" "${inc[@]}" --exclude ".*" --exclude "*/.*" \
    --content-type "$ctype" --cache-control "$cache"
}

upload_tree() { # upload_tree SRC DEST [extra sync args...]
  local src="$1" dest="$2"; shift 2
  typed_sync "$src" "$dest" "$NOCACHE" "text/html; charset=utf-8"        "*.html"
  typed_sync "$src" "$dest" "$SHORT"   "application/json; charset=utf-8" "*.json"
  typed_sync "$src" "$dest" "$SHORT"   "application/x-ndjson; charset=utf-8" "*.jsonl" "*.ndjson"
  typed_sync "$src" "$dest" "$SHORT"   "text/csv; charset=utf-8"         "*.csv"
  typed_sync "$src" "$dest" "$SHORT"   "text/markdown; charset=utf-8"    "*.md"
  typed_sync "$src" "$dest" "$SHORT"   "text/javascript; charset=utf-8"  "*.js" "*.mjs"
  typed_sync "$src" "$dest" "$SHORT"   "text/css; charset=utf-8"         "*.css"
  typed_sync "$src" "$dest" "$LONG"    "image/svg+xml"                   "*.svg"
  typed_sync "$src" "$dest" "$LONG"    "application/wasm"                "*.wasm"
  typed_sync "$src" "$dest" "$LONG"    "application/manifest+json"       "*.webmanifest"
  typed_sync "$src" "$dest" "$LONG"    "image/webp"                      "*.webp"
  typed_sync "$src" "$dest" "$LONG"    "image/avif"                      "*.avif"
  typed_sync "$src" "$dest" "$LONG"    "video/mp4"                       "*.mp4"
  typed_sync "$src" "$dest" "$LONG"    "video/webm"                      "*.webm"
  typed_sync "$src" "$dest" "$LONG"    "font/woff2"                      "*.woff2"
  # Everything else (png/jpg/gif/ico/...) with guessed types, and --delete to drop stale objects.
  # Files uploaded above are newer in S3 than on disk, so this pass skips them.
  aws_ s3 sync "$src" "$dest" --only-show-errors --delete --cache-control "$LONG" \
    --exclude ".*" --exclude "*/.*" --exclude "*.log" --exclude "*.env" --exclude "*.pem" --exclude "*.key" "$@"
}

log "syncing $VIEWER_DIR -> s3://$BUCKET/"
upload_tree "$VIEWER_DIR" "s3://$BUCKET/" --exclude "results/*"

if [[ "$INCLUDE_RESULTS" == "1" && -d "$RESULTS_DIR" ]]; then
  # Public read-only results path for judges: sweep CSVs, run args (JSON), figures. No logs.
  STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
  find "$RESULTS_DIR" -maxdepth 2 -type f \( -name '*.csv' -o -name '*.json' -o -name '*.png' -o -name '*.svg' -o -name '*.md' \) \
    -not -name '.*' -print0 | while IFS= read -r -d '' f; do
      rel="${f#"$RESULTS_DIR"/}"; mkdir -p "$STAGE/$(dirname "$rel")"; cp -p "$f" "$STAGE/$rel"
    done
  log "syncing results ($(find "$STAGE" -type f | wc -l) files) -> s3://$BUCKET/results/"
  upload_tree "$STAGE" "s3://$BUCKET/results/"
fi

log "invalidating CloudFront cache ($DIST_ID)"
INV_ID="$(aws_ cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" --query Invalidation.Id --output text)"
if [[ "$WAIT_FOR_INVALIDATION" == "1" ]]; then
  aws_ cloudfront wait invalidation-completed --distribution-id "$DIST_ID" --id "$INV_ID"
fi

log "done (invalidation $INV_ID)"
echo "$URL"

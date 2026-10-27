#!/usr/bin/env bash
# Publish viewer/ to AWS Amplify Hosting (HTTPS + CDN, manual zip deploy, no IAM role, no build minutes).
# Used because this account cannot create CloudFront distributions until AWS Support verifies it
# (site.yaml remains the preferred S3 + CloudFront path once that is done).
#   bash infra/aws/deploy_amplify.sh          # create the app once, then redeploy viewer/
#   DESTROY=1 bash infra/aws/deploy_amplify.sh   # delete the Amplify app
set -euo pipefail

PROFILE="${AWS_PROFILE:-lost-together-main}"
REGION="${AWS_REGION:-ap-south-1}"
APP_NAME="${APP_NAME:-lost-together}"
BRANCH=main
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
aws_() { aws --profile "$PROFILE" --region "$REGION" "$@"; }

APP_ID=$(aws_ amplify list-apps --query "apps[?name=='$APP_NAME'].appId | [0]" --output text)
if [ "${DESTROY:-0}" = "1" ]; then
  [ "$APP_ID" != "None" ] && aws_ amplify delete-app --app-id "$APP_ID" >/dev/null && echo "deleted Amplify app $APP_ID"
  exit 0
fi

if [ "$APP_ID" = "None" ]; then
  APP_ID=$(aws_ amplify create-app --name "$APP_NAME" --platform WEB \
           --tags project=lost-together --query app.appId --output text)
  aws_ amplify create-branch --app-id "$APP_ID" --branch-name "$BRANCH" >/dev/null
  echo "created Amplify app $APP_ID"
fi

# Security headers for every path (same policy as site.yaml's CloudFront response-headers policy).
HEADERS=$(cat <<'EOF'
customHeaders:
  - pattern: '**'
    headers:
      - key: Content-Security-Policy
        value: "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
      - key: Strict-Transport-Security
        value: max-age=63072000; includeSubDomains
      - key: X-Content-Type-Options
        value: nosniff
      - key: Referrer-Policy
        value: strict-origin-when-cross-origin
EOF
)
aws_ amplify update-app --app-id "$APP_ID" --custom-headers "$HEADERS" >/dev/null

ZIP=$(mktemp -d)/site.zip
(cd "$ROOT/viewer" && python3 -c "
import os, sys, zipfile
with zipfile.ZipFile(sys.argv[1], 'w', zipfile.ZIP_DEFLATED) as z:
    for d, _, fs in os.walk('.'):
        for f in fs:
            z.write(os.path.join(d, f), os.path.relpath(os.path.join(d, f), '.'))
" "$ZIP")
read -r JOB URL < <(aws_ amplify create-deployment --app-id "$APP_ID" --branch-name "$BRANCH" \
                     --query '[jobId, zipUploadUrl]' --output text)
curl -sf -T "$ZIP" "$URL" >/dev/null
aws_ amplify start-deployment --app-id "$APP_ID" --branch-name "$BRANCH" --job-id "$JOB" >/dev/null
for _ in $(seq 1 60); do
  STATUS=$(aws_ amplify get-job --app-id "$APP_ID" --branch-name "$BRANCH" --job-id "$JOB" --query job.summary.status --output text)
  [ "$STATUS" = "SUCCEED" ] && break
  [ "$STATUS" = "FAILED" ] && { echo "deployment failed" >&2; exit 1; }
  sleep 5
done
DOMAIN=$(aws_ amplify get-app --app-id "$APP_ID" --query app.defaultDomain --output text)
echo "https://$BRANCH.$DOMAIN"

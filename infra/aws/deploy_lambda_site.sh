#!/usr/bin/env bash
# Host viewer/ on an AWS Lambda function URL (HTTPS, no CloudFront needed; ~free at demo traffic).
# Used while the account awaits CloudFront verification. The function serves only the files bundled
# into it (allow-list, no path traversal) with the same security headers as site.yaml.
#   bash infra/aws/deploy_lambda_site.sh              # create or update; prints the https URL
#   DESTROY=1 bash infra/aws/deploy_lambda_site.sh    # delete function, URL and role
set -euo pipefail

PROFILE="${AWS_PROFILE:-lost-together-main}"
REGION="${AWS_REGION:-ap-south-1}"
FN=lost-together-viewer
ROLE=lost-together-viewer-role
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
aws_() { aws --profile "$PROFILE" --region "$REGION" "$@"; }

if [ "${DESTROY:-0}" = "1" ]; then
  aws_ lambda delete-function --function-name $FN 2>/dev/null && echo "deleted function $FN" || true
  aws_ iam detach-role-policy --role-name $ROLE --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
  aws_ iam delete-role --role-name $ROLE 2>/dev/null && echo "deleted role $ROLE" || true
  exit 0
fi

ACCOUNT=$(aws_ sts get-caller-identity --query Account --output text)
if ! aws_ iam get-role --role-name $ROLE >/dev/null 2>&1; then
  aws_ iam create-role --role-name $ROLE --tags Key=project,Value=lost-together \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws_ iam attach-role-policy --role-name $ROLE --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  sleep 10   # role propagation
fi

BUILD=$(mktemp -d)
mkdir -p "$BUILD/site"
cp -r "$ROOT/viewer/." "$BUILD/site/"
cat > "$BUILD/app.py" <<'PY'
import base64, mimetypes, os

SITE = os.path.join(os.path.dirname(__file__), "site")
FILES = {os.path.relpath(os.path.join(d, f), SITE).replace(os.sep, "/")
         for d, _, fs in os.walk(SITE) for f in fs}
HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
                               "object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}
TEXT = ("text/", "application/json", "application/javascript", "image/svg+xml")


def handler(event, context):
    path = (event.get("rawPath") or "/").lstrip("/") or "index.html"
    if path.endswith("/"):
        path += "index.html"
    if path not in FILES:                       # allow-list: only bundled files are ever served
        return {"statusCode": 404, "headers": {"Content-Type": "text/plain", **HEADERS}, "body": "not found"}
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(os.path.join(SITE, path), "rb") as f:
        body = f.read()
    headers = {"Content-Type": ctype, "Cache-Control": "max-age=60", **HEADERS}
    if ctype.startswith(TEXT):
        return {"statusCode": 200, "headers": headers, "body": body.decode("utf-8")}
    return {"statusCode": 200, "headers": headers, "body": base64.b64encode(body).decode(), "isBase64Encoded": True}
PY
(cd "$BUILD" && python3 -c "
import os, zipfile
with zipfile.ZipFile('fn.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write('app.py')
    for d, _, fs in os.walk('site'):
        for f in fs:
            z.write(os.path.join(d, f))
")

if aws_ lambda get-function --function-name $FN >/dev/null 2>&1; then
  aws_ lambda update-function-code --function-name $FN --zip-file "fileb://$BUILD/fn.zip" >/dev/null
else
  aws_ lambda create-function --function-name $FN --runtime python3.13 --architectures arm64 --handler app.handler \
    --memory-size 256 --timeout 10 --role "arn:aws:iam::$ACCOUNT:role/$ROLE" --zip-file "fileb://$BUILD/fn.zip" \
    --tags project=lost-together >/dev/null
  aws_ lambda wait function-active-v2 --function-name $FN
  aws_ lambda create-function-url-config --function-name $FN --auth-type NONE >/dev/null
fi
# A public function URL needs both permissions on newer accounts. Re-assert them on every run;
# "already exists" (ResourceConflictException) is fine, any other error is shown.
grant() {
  local out
  if ! out=$(aws_ lambda add-permission --function-name $FN "$@" 2>&1); then
    grep -q ResourceConflictException <<<"$out" || { echo "$out" >&2; return 1; }
  fi
}
grant --statement-id public-url --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE
grant --statement-id public-url-invoke --action lambda:InvokeFunction --principal '*' --invoked-via-function-url
aws_ lambda wait function-updated-v2 --function-name $FN
rm -rf "$BUILD"
URL=$(aws_ lambda get-function-url-config --function-name $FN --query FunctionUrl --output text)
sleep 3
echo "index: $(curl -s -o /dev/null -w '%{http_code}' "$URL")  data/index.json: $(curl -s -o /dev/null -w '%{http_code}' "${URL}data/index.json")"
echo "$URL"

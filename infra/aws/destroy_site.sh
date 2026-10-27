#!/usr/bin/env bash
# Tear down the Lost Together replay site: empty the bucket, delete the stack, wait.
#
#   bash infra/aws/destroy_site.sh          # asks for confirmation
#   bash infra/aws/destroy_site.sh --yes    # non-interactive
# CloudFront distributions take a few minutes to disable before CloudFormation can delete them.
set -euo pipefail

PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-ap-south-1}"
STACK="${STACK_NAME:-lost-together-site}"
EXPECTED_ACCOUNT_SUFFIX="${EXPECTED_ACCOUNT_SUFFIX-8661}"

aws_() { aws --profile "$PROFILE" --region "$REGION" "$@"; }
log() { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

ACCOUNT="$(aws_ sts get-caller-identity --query Account --output text)"
if [[ -n "$EXPECTED_ACCOUNT_SUFFIX" && "$ACCOUNT" != *"$EXPECTED_ACCOUNT_SUFFIX" ]]; then
  die "profile '$PROFILE' is account ...${ACCOUNT: -4}, expected ...$EXPECTED_ACCOUNT_SUFFIX"
fi

if ! aws_ cloudformation describe-stacks --stack-name "$STACK" >/dev/null 2>&1; then
  log "stack $STACK does not exist in $REGION; nothing to do"; exit 0
fi
BUCKET="$(aws_ cloudformation describe-stacks --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text)"

if [[ "${1:-}" != "--yes" ]]; then
  read -r -p "Delete stack '$STACK' and every object in s3://$BUCKET (account ...${ACCOUNT: -4}, $REGION)? Type 'delete': " ans
  [[ "$ans" == "delete" ]] || die "aborted"
fi

if [[ -n "$BUCKET" && "$BUCKET" != "None" ]] && aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  log "emptying s3://$BUCKET"
  aws_ s3 rm "s3://$BUCKET" --recursive --only-show-errors
fi

log "deleting stack $STACK (waits for CloudFront to disable; ~5-15 min)"
aws_ cloudformation delete-stack --stack-name "$STACK"
aws_ cloudformation wait stack-delete-complete --stack-name "$STACK"
log "deleted"

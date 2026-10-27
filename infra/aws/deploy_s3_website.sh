#!/usr/bin/env bash
# Interim hosting for viewer/ as an S3 static website (HTTP only), used until AWS verifies the account
# for CloudFront (then use deploy_site.sh: private bucket + CloudFront OAC + HTTPS + security headers).
# The bucket holds only the public demo files; it is publicly readable and nothing else.
#   bash infra/aws/deploy_s3_website.sh            # create/update and sync
#   DESTROY=1 bash infra/aws/deploy_s3_website.sh  # empty and delete the bucket
set -euo pipefail

PROFILE="${AWS_PROFILE:-lost-together-main}"
REGION="${AWS_REGION:-ap-south-1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
aws_() { aws --profile "$PROFILE" --region "$REGION" "$@"; }
ACCOUNT=$(aws_ sts get-caller-identity --query Account --output text)
BUCKET="lost-together-viewer-$ACCOUNT"

if [ "${DESTROY:-0}" = "1" ]; then
  aws_ s3 rb "s3://$BUCKET" --force && echo "deleted $BUCKET"
  exit 0
fi

if ! aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws_ s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  aws_ s3api put-bucket-tagging --bucket "$BUCKET" --tagging 'TagSet=[{Key=project,Value=lost-together}]'
  aws_ s3api put-bucket-encryption --bucket "$BUCKET" \
    --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
fi
# Public read of objects only (no listing, no writes).
aws_ s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=false,RestrictPublicBuckets=false
aws_ s3api put-bucket-policy --bucket "$BUCKET" --policy "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Sid\":\"PublicReadDemoFiles\",\"Effect\":\"Allow\",\"Principal\":\"*\",\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::$BUCKET/*\"}]}"
aws_ s3api put-bucket-website --bucket "$BUCKET" --website-configuration '{"IndexDocument":{"Suffix":"index.html"}}'
aws_ s3 sync "$ROOT/viewer/" "s3://$BUCKET/" --delete --only-show-errors --cache-control "max-age=60"
echo "http://$BUCKET.s3-website.$REGION.amazonaws.com"

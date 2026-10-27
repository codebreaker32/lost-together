#!/usr/bin/env bash
# Creates the project's IAM developer user and stores its keys as a local AWS profile.
# Run once with an admin profile:  bash infra/iam/create-dev-user.sh [admin-profile] [new-profile] [region]
# Permissions: PowerUserAccess (no IAM/account admin) + IAM limited to project roles (iam-limited.json).
# `cdk bootstrap` attaches AdministratorAccess to its execution role, so run it once with the admin profile.
set -euo pipefail

ADMIN_PROFILE="${1:-bhawna}"
NEW_PROFILE="${2:-lost-together}"
REGION="${3:-ap-south-1}"
USER_NAME=lost-together-dev
GROUP_NAME=lost-together-devs
HERE="$(cd "$(dirname "$0")" && pwd)"

ACCOUNT_ID=$(aws sts get-caller-identity --profile "$ADMIN_PROFILE" --query Account --output text)
POLICY=$(sed "s/ACCOUNT_ID/$ACCOUNT_ID/g" "$HERE/iam-limited.json")

aws iam create-group --group-name "$GROUP_NAME" --profile "$ADMIN_PROFILE" >/dev/null
aws iam attach-group-policy --group-name "$GROUP_NAME" --policy-arn arn:aws:iam::aws:policy/PowerUserAccess --profile "$ADMIN_PROFILE"
aws iam put-group-policy --group-name "$GROUP_NAME" --policy-name lost-together-iam-limited --policy-document "$POLICY" --profile "$ADMIN_PROFILE"
aws iam create-user --user-name "$USER_NAME" --tags Key=project,Value=lost-together --profile "$ADMIN_PROFILE" >/dev/null
aws iam add-user-to-group --user-name "$USER_NAME" --group-name "$GROUP_NAME" --profile "$ADMIN_PROFILE"

# Write the new keys straight into the profile; nothing is echoed.
KEYS=$(aws iam create-access-key --user-name "$USER_NAME" --profile "$ADMIN_PROFILE" --query 'AccessKey.[AccessKeyId,SecretAccessKey]' --output text)
aws configure set aws_access_key_id "$(cut -f1 <<<"$KEYS")" --profile "$NEW_PROFILE"
aws configure set aws_secret_access_key "$(cut -f2 <<<"$KEYS")" --profile "$NEW_PROFILE"
aws configure set region "$REGION" --profile "$NEW_PROFILE"
unset KEYS

aws sts get-caller-identity --profile "$NEW_PROFILE" --query Arn --output text
echo "Profile '$NEW_PROFILE' ready. Consider deactivating the root access keys once you've switched over."

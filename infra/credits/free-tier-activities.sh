#!/usr/bin/env bash
# Completes the two open AWS Free Tier credit activities (+$20 each) and cleans up afterwards.
#   bash infra/credits/free-tier-activities.sh create   [profile] [region]  # Lambda web app + Aurora Serverless v2
#   bash infra/credits/free-tier-activities.sh status   [profile]           # Free Tier activity status
#   bash infra/credits/free-tier-activities.sh cleanup  [profile] [region]  # delete everything this script made
# AWS tracks these as console tutorials; doing the same thing via the API may or may not be credited.
# Run `status` 10-30 minutes after `create`. If an activity is still NOT_STARTED, do that tutorial in the console.
# Aurora bills per ACU-hour while it exists (min capacity 0 lets it auto-pause), so run `cleanup` once credited.
set -euo pipefail

CMD="${1:?usage: create|status|cleanup [profile] [region]}"
PROFILE="${2:-lost-together}"
REGION="${3:-ap-south-1}"
AWS="aws --profile $PROFILE --region $REGION"
FN=lost-together-credit-webapp
ROLE=lost-together-credit-lambda-role
DB=lost-together-credit-db
LAMBDA_ACTIVITY=798b3b7acc72600f4aed6544f1c1e7fb
RDS_ACTIVITY=8205347977b6d2134653bc122ea7d4f3

status() {
  for id in $LAMBDA_ACTIVITY $RDS_ACTIVITY; do
    aws freetier get-account-activity --activity-id "$id" --profile "$PROFILE" --region us-east-1 --query '[title,status]' --output text
  done
  aws freetier get-account-plan-state --profile "$PROFILE" --region us-east-1 --query 'accountPlanRemainingCredits.amount' --output text | sed 's/^/Remaining credits: $/'
}

create_lambda() {
  ACCOUNT_ID=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)
  aws iam create-role --role-name $ROLE --profile "$PROFILE" --tags Key=project,Value=lost-together \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name $ROLE --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole --profile "$PROFILE"
  sleep 10  # let the new role propagate before Lambda uses it

  TMP=$(mktemp -d)
  cat > "$TMP/app.py" <<'PY'
def handler(event, context):
    return {"statusCode": 200, "headers": {"Content-Type": "text/html"},
            "body": "<h1>Lost Together</h1><p>Hello from AWS Lambda.</p>"}
PY
  python3 -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1],'w'); z.write(sys.argv[2],'app.py'); z.close()" "$TMP/fn.zip" "$TMP/app.py"
  $AWS lambda create-function --function-name $FN --runtime python3.13 --architectures arm64 --handler app.handler \
    --role "arn:aws:iam::$ACCOUNT_ID:role/$ROLE" --zip-file "fileb://$TMP/fn.zip" --tags project=lost-together >/dev/null
  $AWS lambda wait function-active-v2 --function-name $FN
  URL=$($AWS lambda create-function-url-config --function-name $FN --auth-type NONE --query FunctionUrl --output text)
  $AWS lambda add-permission --function-name $FN --statement-id public-url --action lambda:InvokeFunctionUrl \
    --principal '*' --function-url-auth-type NONE >/dev/null
  # Newer accounts also require lambda:InvokeFunction for public URLs; ignore if this CLI lacks the flag.
  $AWS lambda add-permission --function-name $FN --statement-id public-url-invoke --action lambda:InvokeFunction \
    --principal '*' --invoked-via-function-url >/dev/null 2>&1 || true
  rm -rf "$TMP"
  sleep 5
  echo "Lambda web app: $URL"
  curl -s -m 10 "$URL" | head -c 200; echo
}

create_aurora() {
  VERSION=$($AWS rds describe-db-engine-versions --engine aurora-postgresql --default-only --query 'DBEngineVersions[0].EngineVersion' --output text)
  echo "Aurora PostgreSQL $VERSION (Serverless v2)"
  $AWS rds create-db-cluster --db-cluster-identifier $DB --engine aurora-postgresql --engine-version "$VERSION" \
    --master-username postgres --manage-master-user-password --storage-encrypted \
    --serverless-v2-scaling-configuration MinCapacity=0,MaxCapacity=1 \
    --tags Key=project,Value=lost-together >/dev/null \
  || $AWS rds create-db-cluster --db-cluster-identifier $DB --engine aurora-postgresql --engine-version "$VERSION" \
    --master-username postgres --manage-master-user-password --storage-encrypted \
    --serverless-v2-scaling-configuration MinCapacity=0.5,MaxCapacity=1 \
    --tags Key=project,Value=lost-together >/dev/null
  $AWS rds create-db-instance --db-instance-identifier $DB-1 --db-cluster-identifier $DB \
    --db-instance-class db.serverless --engine aurora-postgresql --tags Key=project,Value=lost-together >/dev/null
  echo "Waiting for the Aurora instance to become available (5-15 min)..."
  $AWS rds wait db-instance-available --db-instance-identifier $DB-1
  echo "Aurora cluster $DB is available."
}

cleanup() {
  $AWS lambda delete-function --function-name $FN 2>/dev/null && echo "Deleted Lambda $FN" || true
  aws iam detach-role-policy --role-name $ROLE --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole --profile "$PROFILE" 2>/dev/null || true
  aws iam delete-role --role-name $ROLE --profile "$PROFILE" 2>/dev/null && echo "Deleted role $ROLE" || true
  if $AWS rds describe-db-instances --db-instance-identifier $DB-1 >/dev/null 2>&1; then
    $AWS rds wait db-instance-available --db-instance-identifier $DB-1 || true  # can't delete while still creating
    $AWS rds delete-db-instance --db-instance-identifier $DB-1 --skip-final-snapshot >/dev/null
    $AWS rds wait db-instance-deleted --db-instance-identifier $DB-1
  fi
  if $AWS rds describe-db-clusters --db-cluster-identifier $DB >/dev/null 2>&1; then
    $AWS rds delete-db-cluster --db-cluster-identifier $DB --skip-final-snapshot >/dev/null && echo "Deleting Aurora cluster $DB"
  fi
}

case "$CMD" in
  create)  create_lambda; create_aurora; echo; echo "Check credit status in 10-30 min:  bash $0 status $PROFILE" ;;
  create-db) create_aurora ;;
  create-lambda) create_lambda ;;
  status)  status ;;
  cleanup) cleanup ;;
  *) echo "unknown command: $CMD" >&2; exit 1 ;;
esac

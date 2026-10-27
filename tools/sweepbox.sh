#!/usr/bin/env bash
# One c8g.2xlarge (8 Graviton vCPU, ~$0.22/h in ap-south-1) on the DEFAULT account for sweeps.
#   tools/sweepbox.sh up                 # launch (SSH from this machine's IP only, IMDSv2, self-terminates after 4 h)
#   tools/sweepbox.sh sync               # copy the code (no .env/.venv/data/results/.git) and uv sync
#   tools/sweepbox.sh run NAME ARGS...   # start `python -m lost_together.sweep --name NAME ARGS...` in the background
#   tools/sweepbox.sh status             # tail remote sweep logs
#   tools/sweepbox.sh fetch              # pull results/ back
#   tools/sweepbox.sh down               # terminate and delete the key pair and security group
set -euo pipefail

PROFILE="${AWS_PROFILE_SWEEP:-lost-together-main}"
REGION="${AWS_REGION_SWEEP:-ap-south-1}"
TYPE="${SWEEP_INSTANCE_TYPE:-c8g.2xlarge}"
NAME=lost-together-sweepbox
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/.sweepbox"
AWS="aws --profile $PROFILE --region $REGION"
mkdir -p "$STATE"

ssh_() { ssh -i "$STATE/key.pem" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$STATE/known_hosts" \
             -o ConnectTimeout=10 -o ServerAliveInterval=30 "ubuntu@$(cat "$STATE/ip")" "$@"; }

case "${1:?usage: up|sync|run|status|fetch|down}" in
up)
  MYIP=$(curl -s https://checkip.amazonaws.com)
  AMI=$($AWS ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)
  VPC=$($AWS ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
  $AWS ec2 create-key-pair --key-name $NAME --key-type ed25519 --query KeyMaterial --output text > "$STATE/key.pem"
  chmod 600 "$STATE/key.pem"
  SG=$($AWS ec2 create-security-group --group-name $NAME --description "Lost Together sweeps: SSH from team IP" --vpc-id "$VPC" \
       --tag-specifications 'ResourceType=security-group,Tags=[{Key=project,Value=lost-together}]' --query GroupId --output text)
  $AWS ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null
  cat > "$STATE/userdata.sh" <<'EOF'
#!/bin/bash
shutdown -h +240   # dead-man switch: terminates itself after 4 h
apt-get update -qq && apt-get install -y -qq rsync >/dev/null
sudo -u ubuntu -H bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
touch /var/tmp/ready
EOF
  ID=$($AWS ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name $NAME --security-group-ids "$SG" \
       --associate-public-ip-address --instance-initiated-shutdown-behavior terminate --metadata-options HttpTokens=required \
       --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3,DeleteOnTermination=true}' \
       --user-data "file://$STATE/userdata.sh" \
       --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=lost-together}]" \
       --query 'Instances[0].InstanceId' --output text)
  echo "$ID" > "$STATE/id"; echo "$SG" > "$STATE/sg"
  $AWS ec2 wait instance-running --instance-ids "$ID"
  $AWS ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text > "$STATE/ip"
  for _ in $(seq 1 40); do ssh_ test -f /var/tmp/ready 2>/dev/null && break; sleep 10; done
  echo "up: $ID $(cat "$STATE/ip") ($TYPE, $REGION)"
  ;;
sync)
  rsync -az --delete -e "ssh -i $STATE/key.pem -o UserKnownHostsFile=$STATE/known_hosts" \
    --exclude .env --exclude .venv --exclude .git --exclude results --exclude .sweepbox --exclude __pycache__ \
    "$ROOT/" "ubuntu@$(cat "$STATE/ip"):app/"
  ssh_ 'mkdir -p app/results'
  # calibrated perception reads the measured detection tables from results/
  rsync -az -e "ssh -i $STATE/key.pem -o UserKnownHostsFile=$STATE/known_hosts" \
    "$ROOT"/results/calibration.json "$ROOT"/results/calibration_zoom.json "ubuntu@$(cat "$STATE/ip"):app/results/" 2>/dev/null || true
  ssh_ 'cd app && ~/.local/bin/uv sync -q && echo synced'
  ;;
run)
  shift; RUN="$1"; shift
  ssh_ "cd app && nohup ~/.local/bin/uv run python -m lost_together.sweep --name $RUN --workers 8 $* > results/$RUN.log 2>&1; echo \"exit=\$?\" >> results/$RUN.log" </dev/null >/dev/null 2>&1 &
  echo "started $RUN"
  ;;
status)
  ssh_ 'cd app/results && for f in *.log; do echo "== $f: $(tail -1 "$f")"; done'
  ;;
fetch)
  rsync -az -e "ssh -i $STATE/key.pem -o UserKnownHostsFile=$STATE/known_hosts" "ubuntu@$(cat "$STATE/ip"):app/results/" "$ROOT/results/"
  echo "fetched into results/"
  ;;
down)
  ID=$(cat "$STATE/id"); SG=$(cat "$STATE/sg")
  $AWS ec2 terminate-instances --instance-ids "$ID" >/dev/null
  $AWS ec2 wait instance-terminated --instance-ids "$ID"
  $AWS ec2 delete-security-group --group-id "$SG"
  $AWS ec2 delete-key-pair --key-name $NAME
  rm -rf "$STATE"
  echo "terminated $ID and deleted its key pair and security group"
  ;;
esac

#!/usr/bin/env bash
# The agent's model server: one g6e.xlarge (L40S 48 GB, $2.235/h in ap-south-1) on the GPU account,
# Ollama + qwen3-vl:30b-a3b-instruct, reached only through an SSH tunnel (port 11434 is never opened).
# GPU spend is capped at $10 in total (AWS Budget "lost-together-gpu-cap" alerts); keep sessions short.
#   tools/gpubox.sh up        # launch, install Ollama, pull the model (~4 min); self-terminates after MAX_MIN
#   tools/gpubox.sh tunnel    # forward localhost:11434 -> instance (background ssh)
#   tools/gpubox.sh status    # instance state, uptime and spend so far
#   tools/gpubox.sh down      # terminate; delete key pair and security group
set -euo pipefail

PROFILE="${AWS_PROFILE_GPU:-lost-together}"
REGION="${AWS_REGION_GPU:-ap-south-1}"
TYPE="${GPU_INSTANCE_TYPE:-g6e.xlarge}"
PRICE_PER_H="${GPU_PRICE_PER_H:-2.235}"
MAX_MIN="${MAX_MIN:-60}"
MODEL="${OLLAMA_MODEL:-qwen3-vl:30b-a3b-instruct}"   # bare 30b tag = thinking variant
NAME=lost-together-gpubox
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/.gpubox"
AWS="aws --profile $PROFILE --region $REGION"
mkdir -p "$STATE"
ssh_() { ssh -i "$STATE/key.pem" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$STATE/known_hosts" \
             -o ConnectTimeout=10 -o ServerAliveInterval=30 "ubuntu@$(cat "$STATE/ip")" "$@"; }

case "${1:?usage: up|tunnel|status|down}" in
up)
  MYIP=$(curl -s https://checkip.amazonaws.com)
  AMI=$($AWS ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id --query Parameter.Value --output text)
  VPC=$($AWS ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
  SUBNET=$($AWS ec2 describe-subnets --filters Name=vpc-id,Values="$VPC" Name=availability-zone,Values="${REGION}a" --query 'Subnets[0].SubnetId' --output text)
  $AWS ec2 create-key-pair --key-name $NAME --key-type ed25519 --query KeyMaterial --output text > "$STATE/key.pem"
  chmod 600 "$STATE/key.pem"
  SG=$($AWS ec2 create-security-group --group-name $NAME --description "Lost Together model server: SSH from team IP only" --vpc-id "$VPC" \
       --tag-specifications 'ResourceType=security-group,Tags=[{Key=project,Value=lost-together}]' --query GroupId --output text)
  $AWS ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null
  cat > "$STATE/userdata.sh" <<EOF
#!/bin/bash
export HOME=/root
shutdown -h +$MAX_MIN      # dead-man switch (instance-initiated shutdown = terminate)
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable --now ollama
for i in \$(seq 1 60); do curl -s localhost:11434/api/tags >/dev/null && break; sleep 2; done
ollama pull $MODEL && touch /var/tmp/model-ready
EOF
  ID=$($AWS ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name $NAME --security-group-ids "$SG" \
       --subnet-id "$SUBNET" --associate-public-ip-address --instance-initiated-shutdown-behavior terminate \
       --metadata-options HttpTokens=required \
       --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
       --user-data "file://$STATE/userdata.sh" \
       --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=lost-together}]" \
                            'ResourceType=volume,Tags=[{Key=project,Value=lost-together}]' \
       --query 'Instances[0].InstanceId' --output text)
  echo "$ID" > "$STATE/id"; echo "$SG" > "$STATE/sg"; date -u +%s > "$STATE/started"
  $AWS ec2 wait instance-running --instance-ids "$ID"
  $AWS ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text > "$STATE/ip"
  for _ in $(seq 1 60); do ssh_ test -f /var/tmp/model-ready 2>/dev/null && break; sleep 10; done
  ssh_ "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; ollama list | grep -F '$MODEL'"
  echo "up: $ID $(cat "$STATE/ip") ($TYPE); self-terminates after $MAX_MIN min"
  ;;
tunnel)
  pkill -f "ssh .*-L 11434:localhost:11434.*$(cat "$STATE/ip")" 2>/dev/null || true
  ssh -i "$STATE/key.pem" -o UserKnownHostsFile="$STATE/known_hosts" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -fN -L 11434:localhost:11434 "ubuntu@$(cat "$STATE/ip")"
  curl -s localhost:11434/api/tags | python3 -c "import sys,json; print('tunnel ok, models:', [m['name'] for m in json.load(sys.stdin)['models']])"
  ;;
status)
  ID=$(cat "$STATE/id")
  MIN=$(( ($(date -u +%s) - $(cat "$STATE/started")) / 60 ))
  echo "$ID $($AWS ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text), up ${MIN} min, ~\$$(python3 -c "print(round($MIN/60*$PRICE_PER_H, 2))") so far"
  ;;
down)
  ID=$(cat "$STATE/id"); SG=$(cat "$STATE/sg")
  pkill -f "ssh .*-L 11434:localhost:11434" 2>/dev/null || true
  $AWS ec2 terminate-instances --instance-ids "$ID" >/dev/null
  $AWS ec2 wait instance-terminated --instance-ids "$ID"
  $AWS ec2 delete-security-group --group-id "$SG"
  $AWS ec2 delete-key-pair --key-name $NAME
  MIN=$(( ($(date -u +%s) - $(cat "$STATE/started")) / 60 ))
  rm -rf "$STATE"
  echo "terminated $ID after ${MIN} min (~\$$(python3 -c "print(round($MIN/60*$PRICE_PER_H, 2))"))"
  ;;
esac

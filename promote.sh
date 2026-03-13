#!/bin/bash
#
# promote.sh — Push current main to production (playonlineacquire.com)
#
# Usage:
#   bash promote.sh
#
set -e

INSTANCE="i-0bbc6c13fd3dfe6ab"
COMMIT=$(git rev-parse --short HEAD)

echo ""
echo "  Promoting main ($COMMIT) → production"
echo ""

CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "git -C /home/ubuntu/acquire-main -c safe.directory=/home/ubuntu/acquire-main pull origin main",
    "npm --prefix /home/ubuntu/acquire-main install --omit=dev",
    "pm2 restart acquire-main",
    "echo PROD_DEPLOY_DONE"
  ]' \
  --query "Command.CommandId" --output text)

echo "  Waiting..."
sleep 15

STATUS=$(aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE" \
  --query "Status" --output text)

if [ "$STATUS" = "Success" ]; then
  echo "  ✓ Production updated — https://playonlineacquire.com"
else
  echo "  ✗ Deploy failed (status: $STATUS)"
  exit 1
fi
echo ""

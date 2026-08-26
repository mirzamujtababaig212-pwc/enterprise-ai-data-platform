#!/usr/bin/env bash

set -euo pipefail

REGION="us-east-1"
CLUSTER="enterprise-ai-platform-dev"
SERVICE="enterprise-ai-platform-dev-gateway"
ALB_DNS="enterprise-ai-platform-dev-alb-14332853.us-east-1.elb.amazonaws.com"

echo "========================================="
echo "Enterprise AI Platform - DEV Smoke Test"
echo "========================================="

echo
echo "1. ECS service"
echo "-----------------------------------------"

aws ecs describe-services \
  --cluster "$CLUSTER" \
  --services "$SERVICE" \
  --region "$REGION" \
  --query 'services[0].{
    Status:status,
    Desired:desiredCount,
    Running:runningCount,
    Pending:pendingCount
  }'

echo
echo "2. ECS tasks"
echo "-----------------------------------------"

TASKS=$(aws ecs list-tasks \
  --cluster "$CLUSTER" \
  --service-name "$SERVICE" \
  --region "$REGION" \
  --query 'taskArns[]' \
  --output text)

if [[ -z "$TASKS" ]]; then
  echo "ERROR: No ECS tasks found"
  exit 1
fi

aws ecs describe-tasks \
  --cluster "$CLUSTER" \
  --tasks $TASKS \
  --region "$REGION" \
  --query 'tasks[].{
    TaskArn:taskArn,
    LastStatus:lastStatus,
    DesiredStatus:desiredStatus,
    HealthStatus:healthStatus,
    StoppedReason:stoppedReason
  }'

echo
echo "3. SSM parameters"
echo "-----------------------------------------"

aws ssm get-parameters-by-path \
  --path "/enterprise-ai-platform/dev" \
  --region "$REGION" \
  --recursive \
  --query 'Parameters[].{
    Name:Name,
    Type:Type,
    Value:Value
  }' \
  --output table

echo
echo "4. ALB health endpoint"
echo "-----------------------------------------"

curl -fsS \
  -i \
  "http://${ALB_DNS}/healthz"

echo
echo
echo "5. Provider health"
echo "-----------------------------------------"

curl -fsS \
  -i \
  "http://${ALB_DNS}/health"

echo
echo
echo "========================================="
echo "Smoke test completed successfully"
echo "========================================="

resource "aws_iam_role" "sfn_execution" {
  name = "${var.project_name}-${var.environment}-sfn-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "states.amazonaws.com"
        }

        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_iam_role_policy" "sfn_ecs" {
  name = "${var.project_name}-${var.environment}-sfn-ecs"
  role = aws_iam_role.sfn_execution.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Sid    = "ECSRunTask"
        Effect = "Allow"
        Action = [
          "ecs:RunTask"
        ]
        Resource = var.batch_task_definition_arn
      },
      {
        Sid    = "ECSDescribeAndStopTasks"
        Effect = "Allow"
        Action = [
          "ecs:DescribeTasks",
          "ecs:StopTask"
        ]
        Resource = "*"
      },
      {
        Sid    = "PassBatchRoles"
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = [
          var.batch_execution_role_arn,
          var.batch_task_role_arn
        ]
      },
      {
        Sid    = "StepFunctionsEventBridge"
        Effect = "Allow"
        Action = [
          "events:PutRule",
          "events:PutTargets",
          "events:DescribeRule",
        ]
        Resource = "arn:aws:events:${var.aws_region}:*:rule/StepFunctionsGetEventsForECSTaskRule"
      }
    ]
  })
}

resource "aws_sns_topic" "pipeline_notifications" {
  name = "${var.project_name}-${var.environment}-pipeline-notifications"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_iam_role_policy" "sfn_sns" {
  name = "${var.project_name}-${var.environment}-sfn-sns"
  role = aws_iam_role.sfn_execution.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Sid    = "SNSPublishPipelineNotifications"
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = aws_sns_topic.pipeline_notifications.arn
      }
    ]
  })
}

resource "aws_iam_role_policy" "sfn_cloudwatch_logs" {
  name = "${var.project_name}-${var.environment}-sfn-cloudwatch-logs"
  role = aws_iam_role.sfn_execution.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Sid    = "StepFunctionsCloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogDelivery",
          "logs:CreateLogStream",
          "logs:GetLogDelivery",
          "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery",
          "logs:ListLogDeliveries",
          "logs:PutLogEvents",
          "logs:PutResourcePolicy",
          "logs:DescribeResourcePolicies",
          "logs:DescribeLogGroups"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "batch_pipeline" {
  name              = "/aws/vendedlogs/states/${var.project_name}-${var.environment}"
  retention_in_days = 14

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_cloudwatch_metric_alarm" "pipeline_failed" {
  alarm_name          = "${var.project_name}-${var.environment}-batch-pipeline-failed"
  alarm_description   = "Alarm when the AWS Bronze-Silver-Gold Step Functions pipeline fails."
  namespace           = "AWS/States"
  metric_name         = "ExecutionsFailed"
  dimensions = {
    StateMachineArn = aws_sfn_state_machine.batch_pipeline.arn
  }

  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions = [
    aws_sns_topic.pipeline_notifications.arn
  ]

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_cloudwatch_metric_alarm" "pipeline_timed_out" {
  alarm_name          = "${var.project_name}-${var.environment}-batch-pipeline-timed-out"
  alarm_description   = "Alarm when the AWS Bronze-Silver-Gold Step Functions pipeline times out."
  namespace           = "AWS/States"
  metric_name         = "ExecutionsTimedOut"
  dimensions = {
    StateMachineArn = aws_sfn_state_machine.batch_pipeline.arn
  }

  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions = [
    aws_sns_topic.pipeline_notifications.arn
  ]

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_sfn_state_machine" "batch_pipeline" {
  name     = "${var.project_name}-${var.environment}-batch-pipeline"
  role_arn = aws_iam_role.sfn_execution.arn

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.batch_pipeline.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  definition = jsonencode({
    Comment = "Enterprise AI Platform AWS Bronze-Silver-Gold batch pipeline"

    StartAt = "Bronze"

    States = {
      Bronze = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"

        Parameters = {
          Cluster        = var.cluster_arn
          TaskDefinition = var.batch_task_definition_arn
          LaunchType     = "FARGATE"

          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = var.subnet_ids
              SecurityGroups = [var.security_group_id]
              AssignPublicIp = "ENABLED"
            }
          }

          Overrides = {
            ContainerOverrides = [
              {
                Name = "eai-batch"

                Command = [
                  "python",
                  "spark/batch/ingest/batch_to_bronze.py"
                ]
              }
            ]
          }
        }
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            ResultPath   = "$.error"
            Next         = "PipelineFailedNotification"
          }
        ]

        Next = "Silver"
      }

      Silver = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"

        Parameters = {
          Cluster        = var.cluster_arn
          TaskDefinition = var.batch_task_definition_arn
          LaunchType     = "FARGATE"

          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = var.subnet_ids
              SecurityGroups = [var.security_group_id]
              AssignPublicIp = "ENABLED"
            }
          }

          Overrides = {
            ContainerOverrides = [
              {
                Name = "eai-batch"

                Command = [
                  "python",
                  "spark/batch/bronze_to_silver.py"
                ]
              }
            ]
          }
        }
        Retry = [
          {
            ErrorEquals     = ["AmazonECS.Unknown", "States.Timeout"]
            IntervalSeconds = 10
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]

        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            ResultPath   = "$.error"
            Next         = "PipelineFailedNotification"
          }
        ]
        Next = "Gold"
      }

      Gold = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"

        Parameters = {
          Cluster        = var.cluster_arn
          TaskDefinition = var.batch_task_definition_arn
          LaunchType     = "FARGATE"

          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = var.subnet_ids
              SecurityGroups = [var.security_group_id]
              AssignPublicIp = "ENABLED"
            }
          }

          Overrides = {
            ContainerOverrides = [
              {
                Name = "eai-batch"

                Command = [
                  "python",
                  "spark/batch/silver_to_gold.py"
                ]
              }
            ]
          }
        }
        Retry = [
          {
            ErrorEquals     = ["AmazonECS.Unknown", "States.Timeout"]
            IntervalSeconds = 10
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]

        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            ResultPath   = "$.error"
            Next         = "PipelineFailedNotification"
          }
        ]

        Next = "PipelineSucceededNotification"
      }
      PipelineSucceededNotification = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"

        Parameters = {
          TopicArn = aws_sns_topic.pipeline_notifications.arn
          Message  = "Enterprise AI Platform AWS Bronze-Silver-Gold pipeline succeeded."
        }

        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "PipelineSucceeded"
          }
        ]

        Next = "PipelineSucceeded"
      }

      PipelineSucceeded = {
        Type = "Succeed"
      }

      PipelineFailedNotification = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"

        Parameters = {
          TopicArn  = aws_sns_topic.pipeline_notifications.arn
          "Message.$" = "States.JsonToString($)"
        }

        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "PipelineFailed"
          }
        ]

        Next = "PipelineFailed"
      }

      PipelineFailed = {
        Type  = "Fail"
        Error = "EnterpriseAIPipelineFailed"
        Cause = "Bronze, Silver, or Gold ECS batch task failed."
      }
    }
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

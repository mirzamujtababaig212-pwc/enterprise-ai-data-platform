output "state_machine_arn" {
  description = "ARN of the Bronze-Silver-Gold Step Functions state machine."
  value       = aws_sfn_state_machine.batch_pipeline.arn
}

output "execution_role_arn" {
  description = "ARN of the Step Functions execution role."
  value       = aws_iam_role.sfn_execution.arn
}

output "notification_topic_arn" {
  description = "ARN of the SNS topic used for pipeline success and failure notifications."
  value       = aws_sns_topic.pipeline_notifications.arn
}

output "log_group_name" {
  description = "CloudWatch log group used by the Step Functions state machine."
  value       = aws_cloudwatch_log_group.batch_pipeline.name
}

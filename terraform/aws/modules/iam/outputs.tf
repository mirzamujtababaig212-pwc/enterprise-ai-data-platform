output "ecs_execution_role_arn" {
  description = "ARN of the ECS task execution role."
  value       = aws_iam_role.ecs_execution.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role."
  value       = aws_iam_role.ecs_task.arn
}

output "ecs_batch_task_role_arn" {
  description = "ARN of the ECS batch task role."
  value       = aws_iam_role.ecs_batch_task.arn
}

output "ecs_batch_execution_role_arn" {
  description = "ARN of the ECS batch task execution role."
  value       = aws_iam_role.ecs_batch_execution.arn
}
